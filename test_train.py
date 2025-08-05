import pytest
from unittest import mock
from train import main

@pytest.fixture
def mock_args():
    class Args:
        version_fold = "v1"
        ts_arch = "arch"
        LOSS_TYPE = "loss"
        OPT = "adam"
        LR = 0.001
        BATCH_SIZE = 32
        id = "runid"
        resume = False
        saved_model = "/tmp/"
        resume_ckpt = "ckpt.ckpt"
        device = 0
        NUM_EPOCHS = 2
    return Args()

@mock.patch("train.WandbLogger")
@mock.patch("train.TimeSenCLIPLearner")
@mock.patch("train.load_data")
@mock.patch("train.get_callbacks")
@mock.patch("train.pl.Trainer")
def test_main_runs_without_resume(mock_trainer, mock_get_callbacks, mock_load_data, mock_learner, mock_logger, mock_args):
    # Setup mocks
    mock_load_data.return_value = ("train_loader", "val_loader", ["class1", "class2"])
    mock_model = mock.Mock()
    mock_model.parameters.return_value = [mock.Mock(spec=["requires_grad", "size"], requires_grad=True, size=lambda: (2, 2))]
    mock_model.named_parameters.return_value = [("param1", mock.Mock(requires_grad=True))]
    mock_learner.return_value = mock_model
    mock_trainer_instance = mock.Mock()
    mock_trainer.return_value = mock_trainer_instance

    main(mock_args)

    mock_load_data.assert_called_once_with(mock_args)
    mock_logger.assert_called_once()
    mock_learner.assert_called_once_with(args=mock_args, classes=["class1", "class2"])
    mock_trainer.assert_called_once()
    mock_trainer_instance.fit.assert_called_once_with(mock_model, "train_loader", "val_loader")

@mock.patch("train.before_load_weights")
@mock.patch("train.WandbLogger")
@mock.patch("train.TimeSenCLIPLearner")
@mock.patch("train.load_data")
@mock.patch("train.get_callbacks")
@mock.patch("train.pl.Trainer")
def test_main_runs_with_resume(
    mock_trainer, mock_get_callbacks, mock_load_data, mock_learner, mock_logger, mock_before_load_weights, mock_args
):
    # Setup mocks
    mock_args.resume = True
    mock_load_data.return_value = ("train_loader", "val_loader", ["class1"])
    mock_model = mock.Mock()
    mock_model.parameters.return_value = [mock.Mock(spec=["requires_grad", "size"], requires_grad=True, size=lambda: (2, 2))]
    mock_model.named_parameters.return_value = [("param1", mock.Mock(requires_grad=True))]
    mock_learner.return_value = mock_model
    mock_trainer_instance = mock.Mock()
    mock_trainer.return_value = mock_trainer_instance

    main(mock_args)

    mock_before_load_weights.assert_called_once()
    mock_trainer_instance.fit.assert_called_once_with(mock_model, "train_loader", "val_loader")