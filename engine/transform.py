from typing import Any, Protocol, Optional, Dict, List, Tuple, Union
from torchvision.transforms import functional as F
from torchvision import transforms as T
import cv2
import torch
import numpy as np
import math
import random
from PIL import Image

class Transform(Protocol):
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]: ...


class Composition:
    def __init__(self, transformations: List[Transform]) -> None:
        self.transformations = transformations

    def __repr__(self) -> str:
        print("Composition:")
        for i, transformation in enumerate(self.transformations):
            print(f"({i})", transformation)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for transformation in self.transformations:
            data = transformation.transform(data)
        return data


class LoadImg:
    def __init__(self, to_rgb: bool = True) -> None:
        self.to_rgb = to_rgb

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image = cv2.imread(data["img_path"], cv2.IMREAD_UNCHANGED)
        if image.shape[2] == 4:
            image = image[:, :, :3]
        if self.to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # data["img"] = F.to_tensor(image)
        data["img"] = image
        return data

class ToTensor:
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "imgs" in data:
            data['imgs'] = [F.to_tensor(img) for img in data["imgs"]]
        elif "img" in data:
            data["img"] = F.to_tensor(data["img"])
        return data

class NDArrayImgToTensor:
    def __init__(self, to_rgb: bool = True) -> None:
        self.to_rgb = to_rgb

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.to_rgb:
            data["img"] = cv2.cvtColor(data["img"], cv2.COLOR_BGR2RGB)
        data["img"] = F.to_tensor(data["img"])
        return data


class LoadAnn:
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        ann = Image.open(data["ann_path"])
        data["ann"] = torch.from_numpy(np.asarray(ann).copy())[None, :].long()
        return data


class Identity:

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return data


class Resize:
    def __init__(
        self, image_scale: Optional[Tuple[int, int]] = None, antialias: bool = True
    ) -> None:
        self.image_scale = image_scale
        self.antialias = antialias

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "imgs" in data:
            _image_scale = (
                self.image_scale if self.image_scale else data["imgs"][0].shape[-2:]
            )

            data["imgs"] = [F.resize(img, _image_scale, antialias=self.antialias) for img in data["imgs"]]
        elif "img" in data:
            _image_scale = (
                self.image_scale if self.image_scale else data["img"].shape[-2:]
            )

            data["img"] = F.resize(data["img"], _image_scale, antialias=self.antialias)

        if "ann" in data:
            _image_scale = (
                self.image_scale if self.image_scale else data["ann"].shape[-2:]
            )
            data["ann"] = F.resize(
                data["ann"][:, None, :],
                _image_scale,
                interpolation=F.InterpolationMode.NEAREST,
            ).squeeze()

        if "lane_mask" in data:
            _image_scale = (
                self.image_scale if self.image_scale else data["lane_mask"].shape[-2:]
            )
            data["lane_mask"] = F.resize(
                data["lane_mask"].unsqueeze(0).unsqueeze(0),
                _image_scale,
                interpolation=F.InterpolationMode.NEAREST,
            ).squeeze().float()

        return data


class RandomResizeCrop:
    def __init__(
        self,
        image_scale: Tuple[int, int],
        scale: Tuple[float, float],
        crop_size: Tuple[int, int],
        antialias: bool = True,
        # cat_ratio: float = 0.0,
        rare_cat_crop: bool = False,
        patient: int = 10,
        efficient: bool = False,
        efficient_interval: int = 10,
    ) -> None:
        self.image_scale = image_scale
        self.scale = scale
        self.crop_size = np.array(crop_size)
        self.antialias = antialias
        self.rare_cat_crop = rare_cat_crop
        self.patient = patient
        self.efficient = efficient
        if self.efficient:
            self.crop_sizes = np.linspace(self.crop_size, self.crop_size // 2, 6)
            self.efficient_interval = efficient_interval
            self.efficient_counter = 0

    def get_random_size(self):
        min_scale, max_scale = self.scale
        random_scale = random.random() * (max_scale - min_scale) + min_scale
        height = int(self.image_scale[0] * random_scale)
        width = int(self.image_scale[1] * random_scale)
        return height, width

    def get_crop_size(self):
        if self.efficient:
            crop_size = self.crop_sizes[self.efficient_counter // 8]
            self.efficient_counter += 1
            if self.efficient_counter == 48:
                self.efficient_counter = 0
        else:
            crop_size = self.crop_size
        return crop_size.astype(int)

    def get_random_crop(self, scaled_height, scaled_width, crop_size):
        crop_y0 = random.randint(0, scaled_height - crop_size[0])
        crop_x0 = random.randint(0, scaled_width - crop_size[1])

        return crop_y0, crop_x0

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        height, width = self.get_random_size()
        crop_size = self.get_crop_size()
        y0, x0 = self.get_random_crop(height, width, crop_size)
        if "ann" in data:
            if self.rare_cat_crop:
                assert (
                    "ann" in data
                ), "Category-ratio cropping is avaliable only when label is given!"
                if "random_cat_id" in data:
                    random_id = data["random_cat_id"]
                else:
                    random_id = random.choice(
                        data["ann"].unique(sorted=False)
                    )  # Choose a random category id in the label
                uncropped_ann = F.resize(
                    data["ann"][:, None, :],
                    (height, width),
                    interpolation=F.InterpolationMode.NEAREST,
                ).squeeze()

                best_ratio = 0
                best_x0 = x0
                best_y0 = y0
                for _ in range(self.patient):
                    ann = uncropped_ann[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]
                    ratio = (
                        torch.where(ann == random_id)[0].shape[0]
                        / ann.flatten().shape[0]
                    )
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_x0 = x0
                        best_y0 = y0

                    y0, x0 = self.get_random_crop(height, width)

                x0 = best_x0
                y0 = best_y0
                data["ann"] = uncropped_ann[
                    y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]
                ]
            else:
                data["ann"] = F.resize(
                    data["ann"][:, None, :],
                    (height, width),
                    interpolation=F.InterpolationMode.NEAREST,
                ).squeeze()[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]

        if "imgs" in data:
            data["imgs"] = [F.resize(
                img, (height, width), antialias=self.antialias
            )[:, y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]] for img in data["imgs"]]
        elif "img" in data:
            data["img"] = F.resize(
                data["img"], (height, width), antialias=self.antialias
            )[:, y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]

        data["height"] = height
        data["width"] = width
        data["y0"] = y0
        data["x0"] = x0

        return data


class Normalize:
    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.mean = mean
        self.std = std

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "imgs" in data:
            data["imgs"] = [F.normalize(img, self.mean, self.std) for img in data["imgs"]]
        elif "img" in data:
            data["img"] = F.normalize(data["img"], self.mean, self.std)

        return data


class ColorJitter:

    def __init__(
        self,
        brightness: Union[float, Tuple[float, float]] = 0,
        contrast: Union[float, Tuple[float, float]] = 0,
        saturation: Union[float, Tuple[float, float]] = 0,
        hue: Union[float, Tuple[float, float]] = 0,
    ) -> None:
        self.jitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "imgs" in data:
            data["imgs"] = [self.jitter(img) for img in data["imgs"]]
        elif "img" in data:
            data["img"] = self.jitter(data["img"])

        return data


class WeakAndStrong:
    def __init__(self, weak_transform: Transform, strong_transform: Transform) -> None:
        self.weak = weak_transform
        self.strong = strong_transform

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "imgs" in data:
            weak_imgs = self.weak.transform(data)["imgs"]
            data["strong_imgs"] = self.strong.transform(data)["imgs"]
            data["imgs"] = weak_imgs
        elif "img" in data:
            weak_img = self.weak.transform(data)["img"]
            data["strong_img"] = self.strong.transform(data)["img"]
            data["img"] = weak_img

        return data


class RandomGaussian:
    def __init__(self, p: float = 0.5, kernel_size: int = 3) -> None:
        self.p = p
        self.kernel_size = kernel_size

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() >= self.p:
            if "imgs" in data:
                data["imgs"] = [F.gaussian_blur(img, self.kernel_size) for img in data["imgs"]]
            elif "img" in data:
                data["img"] = F.gaussian_blur(data["img"], self.kernel_size)

        return data


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() > self.p:
            if "imgs" in data:
                data["imgs"] = [F.hflip(img) for img in data["imgs"]]
            elif "img" in data:
                data["img"] = F.hflip(data["img"])

            if "ann" in data:
                data["ann"] = F.hflip(data["ann"][:, None, :])[:, 0]

        return data


class RandomErase:
    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: int = 0,
    ) -> None:
        self.erase = T.RandomErasing(p, scale, ratio, value)

    def get_params(self, img: torch.Tensor) -> Tuple[int, int, int, int, torch.Tensor]:
        # cast self.value to script acceptable type
        if isinstance(self.erase.value, (int, float)):
            value = [float(self.erase.value)]
        elif isinstance(self.erase.value, str):
            value = None
        elif isinstance(self.erase.value, (list, tuple)):
            value = [float(v) for v in self.erase.value]
        else:
            value = self.erase.value

        if value is not None and not (len(value) in (1, img.shape[-3])):
            raise ValueError(
                "If value is a sequence, it should have either a single value or "
                f"{img.shape[-3]} (number of input channels)"
            )
        
        x, y, h, w, v =  self.erase.get_params(img, scale=self.erase.scale, ratio=self.erase.ratio, value=value)
        return x, y, h, w, v

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # 只在第一次呼叫時初始化 erased_imgs（避免後續呼叫重設，讓多個 erase 累積）
        if data["domain"] == 1 and "erased imgs" not in data and "erased img" not in data:
            if "imgs" in data:
                data["erased imgs"] = [img.clone() for img in data["imgs"]]
            elif "img" in data:
                data["erased img"] = data["img"].clone()
        
        img_p = data["imgs"][0] if "imgs" in data else data["img"]
        x, y, h, w, v = self.get_params(img_p)
        p = torch.rand(1)
        if "erased img" in data or "erased imgs" in data:
            if p <= self.erase.p:
                return data
            if "erased imgs" in data:
                data["erased imgs"] = [F.erase(img, x, y, h, w, v, self.erase.inplace) for img in data["erased imgs"]]
            elif "erased img" in data:
                data["erased img"] = F.erase(data["erased img"], x, y, h, w, v, self.erase.inplace)
        else:
            if p > self.erase.p:
                if "imgs" in data:
                    data["erased imgs"] = [F.erase(img, x, y, h, w, v, self.erase.inplace) for img in data["imgs"]]
                elif "img" in data:
                    data["erased img"] = F.erase(data["img"], x, y, h, w, v, self.erase.inplace)
            else:
                if "imgs" in data:
                    data["erased imgs"] = data["imgs"].copy()
                elif "img" in data:
                    data["erased img"] = data["img"].clone()

        return data

class ContrastStretch:
    def __init__(
        self,
        max_intensity: float,
        function_name: str,
        parameter
    ) -> None:
        self.max_intensity = max_intensity
        self.function_name = function_name
        self.parameter = parameter
    
    def sigmoid(self, values, power):
        xp = np.power(values, power)
        xip = np.power(1.0 - values, power)
        return xp / (xp + xip)
    
    def asym_sigmoid(self, values, params: Tuple[float, float]):
        power, pivot = params
        y = values.copy()
        y[values <= pivot] = pivot * np.power(values[values <= pivot] / pivot, power)
        rpivot = 1.0 - pivot
        y[values > pivot] = 1.0 - rpivot * np.power((1.0 - values[values > pivot]) / rpivot, power)
        return y
    
    def log(self, value, alpha):
        y = np.log1p(alpha * value)
        y = y / np.max(y)
        return y
    
    def exp(self, value, power):
        y = np.power(value, power)
        return y
    
    def gamma_correction(self, value, gamma):
        y = np.power(value, 1 / gamma)
        return y

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        func = getattr(self, self.function_name, None)
        assert callable(func), f'There is no {self.function_name} transform in transform.py!'

        max_value = self.max_intensity if self.max_intensity != 0.0 else data["img"].astype(np.float32).mean() * 8.0
        img = data["img"].astype(np.float32) / max_value
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        np.clip(img[:, :, 2], 0, 1, out=img[:, :, 2])   # ensure there's no exceeded value.

        self.parameter = tuple(self.parameter) if type(self.parameter) == list else self.parameter
        img[:, :, 2] = func(img[:, :, 2], self.parameter)

        # data.setdefault("imgs", list()).append((cv2.cvtColor(img, cv2.COLOR_HSV2RGB) * 255).astype(np.uint8))
        data.setdefault("imgs", list()).append((cv2.cvtColor(img, cv2.COLOR_HSV2RGB)).astype(np.float32))

        return data
    
class Check:
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "imgs" in data and "img" in data:
            data.pop("img")
        return data