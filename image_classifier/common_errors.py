class CheckpointFilePathNotSpecifiedError(TypeError):
    def __init__(self):
        super().__init__(
            "Checkpoint (`testing.checkpoint_to_test_path` setting) "
            + "must be specified in the config"
        )
