"""
Alert Management System
===================

.. module:: pipeline.monitoring.alerts
   :synopsis: Alert management and notification system

.. moduleauthor:: aai540-group3
"""

import smtplib
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional, Set

import requests
from jinja2 import Template
from omegaconf import DictConfig

from ..utils.logging import get_logger

logger = get_logger(__name__)


class AlertManager:
    """Manage and distribute system alerts.

    :param cfg: Alert configuration
    :type cfg: DictConfig
    :raises ValueError: If configuration is invalid
    """

    def __init__(self, cfg: DictConfig):
        """Initialize alert manager.

        :param cfg: Alert configuration
        :type cfg: DictConfig
        """
        self.cfg = cfg
        self.active_alerts: List[Dict] = []
        self.alert_history: List[Dict] = []
        self._load_templates()

    def _load_templates(self) -> None:
        """Load alert templates."""
        self.templates = {
            "slack": Template(
                """
                *{{ alert.severity | upper }} Alert*
                *Type:* {{ alert.type }}
                *Message:* {{ alert.message }}
                *Time:* {{ alert.timestamp }}
                {% if alert.metadata %}
                *Details:*
                {% for key, value in alert.metadata.items() %}
                • {{ key }}: {{ value }}
                {% endfor %}
                {% endif %}
            """.strip()
            ),
            "email": Template(
                """
                <h2>{{ alert.severity | upper }} Alert</h2>
                <p><strong>Type:</strong> {{ alert.type }}</p>
                <p><strong>Message:</strong> {{ alert.message }}</p>
                <p><strong>Time:</strong> {{ alert.timestamp }}</p>
                {% if alert.metadata %}
                <h3>Details:</h3>
                <ul>
                {% for key, value in alert.metadata.items() %}
                    <li><strong>{{ key }}:</strong> {{ value }}</li>
                {% endfor %}
                </ul>
                {% endif %}
            """.strip()
            ),
        }

    def create_alert(
        self,
        alert_type: str,
        message: str,
        severity: str,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """Create new alert.

        :param alert_type: Type of alert
        :type alert_type: str
        :param message: Alert message
        :type message: str
        :param severity: Alert severity
        :type severity: str
        :param metadata: Additional metadata
        :type metadata: Optional[Dict]
        :return: Created alert
        :rtype: Dict
        """
        alert = {
            "id": self._generate_alert_id(),
            "type": alert_type,
            "message": message,
            "severity": severity,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "status": "active",
            "acknowledged": False,
        }

        self.active_alerts.append(alert)
        self.alert_history.append(alert)

        # Send notifications
        self._send_notifications(alert)

        return alert

    def acknowledge_alert(self, alert_id: str) -> None:
        """Acknowledge an alert.

        :param alert_id: Alert ID
        :type alert_id: str
        :raises ValueError: If alert not found
        """
        for alert in self.active_alerts:
            if alert["id"] == alert_id:
                alert["acknowledged"] = True
                alert["acknowledged_at"] = datetime.now().isoformat()
                return

        raise ValueError(f"Alert not found: {alert_id}")

    def resolve_alert(self, alert_id: str, resolution: str) -> None:
        """Resolve an alert.

        :param alert_id: Alert ID
        :type alert_id: str
        :param resolution: Resolution description
        :type resolution: str
        :raises ValueError: If alert not found
        """
        for alert in self.active_alerts:
            if alert["id"] == alert_id:
                alert["status"] = "resolved"
                alert["resolution"] = resolution
                alert["resolved_at"] = datetime.now().isoformat()
                self.active_alerts.remove(alert)
                return

        raise ValueError(f"Alert not found: {alert_id}")

    def _send_notifications(self, alert: Dict) -> None:
        """Send alert notifications.

        :param alert: Alert to send
        :type alert: Dict
        """
        if self._should_send_notification(alert):
            if self.cfg.slack.enabled:
                self._send_slack_notification(alert)
            if self.cfg.email.enabled:
                self._send_email_notification(alert)
            if self.cfg.pagerduty.enabled:
                self._send_pagerduty_notification(alert)

    def _should_send_notification(self, alert: Dict) -> bool:
        """Check if notification should be sent.

        :param alert: Alert to check
        :type alert: Dict
        :return: Whether to send notification
        :rtype: bool
        """
        # Check notification rules
        rules = self.cfg.notification_rules

        # Check severity threshold
        if alert["severity"] not in rules.severity_levels:
            return False

        # Check rate limiting
        if not self._check_rate_limit(alert):
            return False

        # Check time window
        if not self._check_time_window(alert):
            return False

        return True

    def _send_slack_notification(self, alert: Dict) -> None:
        """Send Slack notification.

        :param alert: Alert to send
        :type alert: Dict
        """
        try:
            webhook_url = self.cfg.slack.webhook_url
            if not webhook_url:
                logger.warning("Slack webhook URL not configured")
                return

            message = self.templates["slack"].render(alert=alert)
            response = requests.post(webhook_url, json={"text": message}, timeout=5)
            response.raise_for_status()

        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")

    def _send_email_notification(self, alert: Dict) -> None:
        """Send email notification.

        :param alert: Alert to send
        :type alert: Dict
        """
        try:
            if not self.cfg.email.recipients:
                logger.warning("Email recipients not configured")
                return

            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"{alert['severity'].upper()} Alert: {alert['type']}"
            msg["From"] = self.cfg.email.sender
            msg["To"] = ", ".join(self.cfg.email.recipients)

            html_content = self.templates["email"].render(alert=alert)
            msg.attach(MIMEText(html_content, "html"))

            with smtplib.SMTP(
                self.cfg.email.smtp_host, self.cfg.email.smtp_port
            ) as server:
                if self.cfg.email.use_tls:
                    server.starttls()
                if self.cfg.email.username and self.cfg.email.password:
                    server.login(self.cfg.email.username, self.cfg.email.password)
                server.send_message(msg)

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")

    def _send_pagerduty_notification(self, alert: Dict) -> None:
        """Send PagerDuty notification.

        :param alert: Alert to send
        :type alert: Dict
        """
        try:
            if not self.cfg.pagerduty.service_key:
                logger.warning("PagerDuty service key not configured")
                return

            payload = {
                "service_key": self.cfg.pagerduty.service_key,
                "event_type": "trigger",
                "description": alert["message"],
                "client": "MLOps Pipeline",
                "client_url": self.cfg.base_url,
                "details": alert,
            }

            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue", json=payload, timeout=5
            )
            response.raise_for_status()

        except Exception as e:
            logger.error(f"Failed to send PagerDuty notification: {e}")

    def get_active_alerts(
        self, severity: Optional[str] = None, alert_type: Optional[str] = None
    ) -> List[Dict]:
        """Get active alerts with optional filtering.

        :param severity: Filter by severity
        :type severity: Optional[str]
        :param alert_type: Filter by type
        :type alert_type: Optional[str]
        :return: List of active alerts
        :rtype: List[Dict]
        """
        alerts = self.active_alerts

        if severity:
            alerts = [a for a in alerts if a["severity"] == severity]
        if alert_type:
            alerts = [a for a in alerts if a["type"] == alert_type]

        return alerts

    def get_alert_history(
        self, days: Optional[int] = None, include_resolved: bool = True
    ) -> List[Dict]:
        """Get alert history.

        :param days: Number of days of history
        :type days: Optional[int]
        :param include_resolved: Include resolved alerts
        :type include_resolved: bool
        :return: Alert history
        :rtype: List[Dict]
        """
        alerts = self.alert_history

        if not include_resolved:
            alerts = [a for a in alerts if a["status"] != "resolved"]

        if days:
            cutoff = datetime.now() - timedelta(days=days)
            alerts = [
                a for a in alerts if datetime.fromisoformat(a["timestamp"]) > cutoff
            ]

        return alerts

    def cleanup_old_alerts(self) -> None:
        """Clean up old alerts based on retention policy."""
        retention_days = self.cfg.retention_days
        cutoff = datetime.now() - timedelta(days=retention_days)

        self.alert_history = [
            alert
            for alert in self.alert_history
            if datetime.fromisoformat(alert["timestamp"]) > cutoff
        ]
