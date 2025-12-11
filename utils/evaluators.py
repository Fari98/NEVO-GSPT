from sklearn.metrics import matthews_corrcoef
import torch

def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.

    Returns
    -------
    torch.Tensor
        RMSE value.
    """
    return torch.sqrt(torch.mean(torch.square(torch.sub(y_true, y_pred)), dim=len(y_pred.shape) - 1)).item()

def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.

    Returns
    -------
    torch.Tensor
        RMSE value.
    """
    return torch.mean(torch.square(torch.sub(y_true, y_pred)), dim=len(y_pred.shape) - 1).item()



def modified_sigmoid( scaling_factor = 1):
        def ms(tensor):
            return torch.div(1,torch.add(1,torch.exp(torch.mul(-1,torch.mul(tensor, scaling_factor)))))

        return ms

def binarized_rmse(binarizer = modified_sigmoid()):
    def brmse(y_true, y_pred):
        my_pred = binarizer(y_pred)
        return torch.sqrt(torch.mean(torch.pow(torch.sub(y_true, my_pred), 2), len(y_pred.shape) - 1)).item()

    return brmse


def binarized_bce(binarizer=modified_sigmoid()):
    """
    Binary Cross Entropy evaluator with binarization.

    Uses the provided binarizer to convert predictions to binary values before
    computing Binary Cross Entropy loss.

    Parameters
    ----------
    binarizer : callable
        Function that binarizes predictions. Default: modified_sigmoid()

    Returns
    -------
    callable
        Function that computes binarized BCE loss.
    """
    def bce(y_true, y_pred):
        my_pred = binarizer(y_pred)
        # Clamp predictions to avoid log(0)
        my_pred = torch.clamp(my_pred, min=1e-7, max=1-1e-7)
        bce_loss = torch.mean(
            y_true * torch.log(my_pred) + (1 - y_true) * torch.log(1 - my_pred),
            dim=len(y_pred.shape) - 1
        )
        return -bce_loss.item()

    return bce


def binarized_mcc(binarizer=modified_sigmoid()):
    """
    Matthews Correlation Coefficient evaluator with binarization.

    Uses the provided binarizer to convert predictions to binary values before
    computing MCC using scikit-learn.

    Parameters
    ----------
    binarizer : callable
        Function that binarizes predictions. Default: modified_sigmoid()

    Returns
    -------
    callable
        Function that computes MCC score.
    """


    def mcc(y_true, y_pred):
        my_pred = binarizer(y_pred)
        # Convert to binary (round to 0 or 1)
        my_pred_binary = torch.round(my_pred).detach().cpu().numpy().flatten().astype(int)
        y_true_binary = y_true.detach().cpu().numpy().flatten().astype(int)

        mcc_score = matthews_corrcoef(y_true_binary, my_pred_binary)
        return -mcc_score
    return mcc


def cross_entropy(y_true, y_pred):
    return torch.nn.functional.cross_entropy(y_pred, y_true.long()).item()