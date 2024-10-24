import hydra


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    connection = hydra.utils.instantiate(cfg.db)
    connection.connect()


if __name__ == "__main__":
    main()
