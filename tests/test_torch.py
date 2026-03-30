"""Tests for arrayview._torch (PyTorch DL integration)."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock


class FakeTensor:
    """Mimics a PyTorch tensor with .detach().cpu().numpy() chain."""
    def __init__(self, arr):
        self._arr = np.asarray(arr)
    def detach(self):
        return self
    def cpu(self):
        return self
    def numpy(self):
        return self._arr


class TestTensorToNdarray:
    def test_numpy_passthrough(self):
        from arrayview._torch import _tensor_to_ndarray
        arr = np.zeros((4, 8, 8))
        assert _tensor_to_ndarray(arr) is arr

    def test_fake_tensor(self):
        from arrayview._torch import _tensor_to_ndarray
        t = FakeTensor(np.ones((2, 3)))
        result = _tensor_to_ndarray(t)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result, 1.0)


class TestExtractImages:
    def test_ndarray_batch(self):
        from arrayview._torch import _extract_images
        batch = np.random.rand(4, 64, 64)
        result = _extract_images(batch)
        assert result.shape == (4, 64, 64)

    def test_tensor_batch(self):
        from arrayview._torch import _extract_images
        batch = FakeTensor(np.random.rand(4, 64, 64))
        result = _extract_images(batch)
        assert result.shape == (4, 64, 64)

    def test_dict_batch_auto_key(self):
        from arrayview._torch import _extract_images
        batch = {
            'label': np.zeros((4, 64, 64), dtype=np.int32),
            'image': np.random.rand(4, 1, 128, 128).astype(np.float32),
        }
        result = _extract_images(batch)
        assert result.shape == (4, 1, 128, 128)

    def test_dict_batch_explicit_key(self):
        from arrayview._torch import _extract_images
        batch = {
            'image': np.random.rand(4, 128, 128).astype(np.float32),
            'label': np.zeros((4, 64, 64), dtype=np.int32),
        }
        result = _extract_images(batch, key='label')
        assert result.shape == (4, 64, 64)

    def test_tuple_batch(self):
        from arrayview._torch import _extract_images
        batch = (np.random.rand(4, 64, 64), np.zeros(4))
        result = _extract_images(batch)
        assert result.shape == (4, 64, 64)

    def test_unsupported_type(self):
        from arrayview._torch import _extract_images
        with pytest.raises(TypeError):
            _extract_images("not a batch")


class TestViewBatch:
    @patch('arrayview._torch.view')
    def test_ndarray_batch(self, mock_view):
        from arrayview._torch import view_batch
        mock_view.return_value = MagicMock()
        batch = np.random.rand(4, 64, 64)
        view_batch(batch)
        mock_view.assert_called_once()
        arr_arg = mock_view.call_args[0][0]
        assert arr_arg.shape == (4, 64, 64)

    @patch('arrayview._torch.view')
    def test_dict_batch_with_overlay(self, mock_view):
        from arrayview._torch import view_batch
        mock_view.return_value = MagicMock()
        batch = {
            'image': np.random.rand(4, 64, 64).astype(np.float32),
            'label': np.ones((4, 64, 64), dtype=np.int32),
        }
        view_batch(batch, overlay='label')
        mock_view.assert_called_once()
        kwargs = mock_view.call_args[1]
        assert kwargs['overlay'].shape == (4, 64, 64)

    @patch('arrayview._torch.view')
    def test_dataloader(self, mock_view):
        from arrayview._torch import view_batch
        mock_view.return_value = MagicMock()
        batch = np.random.rand(8, 32, 32)
        class FakeLoader:
            dataset = True  # has 'dataset' attr like real DataLoader
            def __iter__(self):
                return iter([batch])
        view_batch(FakeLoader())
        mock_view.assert_called_once()

    @patch('arrayview._torch.view')
    def test_dataset_with_samples(self, mock_view):
        from arrayview._torch import view_batch
        mock_view.return_value = MagicMock()
        class FakeDataset:
            def __len__(self):
                return 100
            def __getitem__(self, idx):
                return np.random.rand(64, 64)
        view_batch(FakeDataset(), samples=4)
        mock_view.assert_called_once()
        arr_arg = mock_view.call_args[0][0]
        assert arr_arg.shape == (4, 64, 64)


# ---------- TrainingMonitor ----------

class TestTrainingMonitor:
    @patch('arrayview._torch.view')
    def test_first_step_opens_viewer(self, mock_view):
        from arrayview._torch import TrainingMonitor
        handle = MagicMock()
        mock_view.return_value = (handle, handle, handle)

        mon = TrainingMonitor(every=1, samples=1)
        mon.step(
            input=np.random.rand(1, 64, 64),
            target=np.ones((1, 64, 64), dtype=np.int32),
            prediction=np.random.rand(1, 64, 64),
            epoch=0,
        )
        mock_view.assert_called_once()

    @patch('arrayview._torch.view')
    def test_skipped_epoch(self, mock_view):
        from arrayview._torch import TrainingMonitor
        handle = MagicMock()
        mock_view.return_value = (handle, handle, handle)

        mon = TrainingMonitor(every=5, samples=1)
        # epoch 0 — opens viewer
        mon.step(input=np.zeros((1, 8, 8)), target=np.zeros((1, 8, 8)),
                 prediction=np.zeros((1, 8, 8)), epoch=0)
        mock_view.assert_called_once()

        # epoch 1 — skipped (not divisible by 5, not epoch 0)
        handle.reset_mock()
        mon.step(input=np.zeros((1, 8, 8)), target=np.zeros((1, 8, 8)),
                 prediction=np.zeros((1, 8, 8)), epoch=1)
        handle.update.assert_not_called()

    @patch('arrayview._torch.view')
    def test_update_called_on_matching_epoch(self, mock_view):
        from arrayview._torch import TrainingMonitor
        handle = MagicMock()
        mock_view.return_value = (handle, handle, handle)

        mon = TrainingMonitor(every=5, samples=1)
        mon.step(input=np.zeros((1, 8, 8)), target=np.zeros((1, 8, 8)),
                 prediction=np.zeros((1, 8, 8)), epoch=0)

        # epoch 5 — should update
        mon.step(input=np.ones((1, 8, 8)), target=np.ones((1, 8, 8)),
                 prediction=np.ones((1, 8, 8)), epoch=5)
        assert handle.update.call_count == 3  # input, target, prediction handles

    @patch('arrayview._torch.view')
    def test_two_pane_mode(self, mock_view):
        from arrayview._torch import TrainingMonitor
        handle = MagicMock()
        mock_view.return_value = (handle, handle)

        mon = TrainingMonitor(every=1, samples=1)
        mon.step(input=np.zeros((1, 8, 8)), prediction=np.zeros((1, 8, 8)), epoch=0)
        assert mock_view.call_count == 1
        # Should have been called with 2 arrays
        args = mock_view.call_args[0]
        assert len(args) == 2

    @patch('arrayview._torch.view')
    def test_single_pane_mode(self, mock_view):
        from arrayview._torch import TrainingMonitor
        handle = MagicMock()
        mock_view.return_value = handle

        mon = TrainingMonitor(every=1, samples=1)
        mon.step(prediction=np.zeros((1, 8, 8)), epoch=0)
        assert mock_view.call_count == 1
        args = mock_view.call_args[0]
        assert len(args) == 1

    @patch('arrayview._torch.view')
    def test_multiple_samples_stacked(self, mock_view):
        from arrayview._torch import TrainingMonitor
        handle = MagicMock()
        mock_view.return_value = (handle, handle, handle)

        mon = TrainingMonitor(every=1, samples=3)
        # Call step 3 times in same epoch → collects 3 samples
        for i in range(3):
            mon.step(
                input=np.full((1, 8, 8), i, dtype=np.float32),
                target=np.zeros((1, 8, 8), dtype=np.int32),
                prediction=np.full((1, 8, 8), i * 0.1, dtype=np.float32),
                epoch=0,
            )
        # view() called once, after samples collected, with stacked arrays
        assert mock_view.call_count == 1
        arr_arg = mock_view.call_args[0][0]
        assert arr_arg.shape[0] == 3  # 3 samples stacked
