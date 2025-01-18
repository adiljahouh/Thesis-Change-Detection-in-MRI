import torch
import kornia.geometry.transform as kornia_transform
# class ShiftImage:
#     def __init__(self, max_shift_x, max_shift_y):
#         self.max_shift_x = max_shift_x
#         self.max_shift_y = max_shift_y

#     def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
#         # Ensure the tensor is 3D
#         shift_x = torch.randint(-self.max_shift_x, self.max_shift_x + 1, (1,)).double()
#         shift_y = torch.randint(-self.max_shift_y, self.max_shift_y + 1, (1,)).double()

#         # Create translation tensor
#         translation = torch.tensor([[shift_x.item(), shift_y.item()]]).double()
#         return kornia_transform.translate(tensor.unsqueeze(0).double(), translation, mode='bilinear', padding_mode='border', align_corners=True).squeeze(0).float()
class ShiftImage:
    def __init__(self, max_shift_x, max_shift_y):
        self.max_shift_x = max_shift_x
        self.max_shift_y = max_shift_y

    def __call__(self, tensor: torch.Tensor, shift: tuple = None) -> torch.Tensor:
        # If no shift is provided, raise an error
        if shift is None:
            raise ValueError("Shift values must be explicitly provided.")

        shift_x, shift_y = shift
        translation = torch.tensor([[shift_x, shift_y]], dtype=torch.double)

        return kornia_transform.translate(
            tensor.unsqueeze(0).double(),
            translation,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(0).float()

class RotateImage:
    def __init__(self, angle, center=None, mode='bilinear', padding_mode='zeros', align_corners=True):
        self.angle = angle  # The rotation angle in degrees
        self.center = center
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def __call__(self, img):
        # Ensure the input is a tensor
        if not isinstance(img, torch.Tensor):
            raise TypeError("Input must be a tensor")
        
        # # Add a batch dimension if needed (Kornia expects a batch of tensors)
        # if img.ndim == 3:  # [C, H, W]
        #     img = img.unsqueeze(0)  # Add batch dimension [B, C, H, W]
        
        return kornia_transform.rotate(
            img, self.angle, self.center, self.mode, self.padding_mode, self.align_corners
        )
