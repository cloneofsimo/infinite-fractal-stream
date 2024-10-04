import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def fractal_kernel(
    zr_ptr,
    zi_ptr,
    cr_ptr,
    ci_ptr,
    output_ptr,
    alpha_ptr,
    beta_ptr,
    poly0_ptr,
    poly1_ptr,
    poly2_ptr,
    poly3_ptr,
    frequency_ptr,
    p_ptr,
    R,
    max_iter,
    H,
    W,
    fractal_type,
    BLOCK_SIZE: tl.constexpr,
):
    bid = tl.program_id(0)
    pid = tl.program_id(1)
    grid_offset = pid * BLOCK_SIZE
    offsets = grid_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < H * W

    batch_offset = bid * 3 * H * W
    output_ptr += batch_offset

    zr_ptr += bid * H * W
    zi_ptr += bid * H * W

    alpha = tl.load(alpha_ptr + bid)
    beta = tl.load(beta_ptr + bid)
    p = tl.load(p_ptr + bid)

    poly0 = tl.load(poly0_ptr + bid)
    poly1 = tl.load(poly1_ptr + bid)
    poly2 = tl.load(poly2_ptr + bid)
    poly3 = tl.load(poly3_ptr + bid)
    frequency = tl.load(frequency_ptr + bid)
    zr = tl.load(zr_ptr + offsets, mask=mask)
    zi = tl.load(zi_ptr + offsets, mask=mask)
    cr = tl.load(cr_ptr + offsets, mask=mask)
    ci = tl.load(ci_ptr + offsets, mask=mask)

    iteration = tl.zeros([BLOCK_SIZE], dtype=tl.int8)

    for i in range(max_iter):
        if fractal_type == 0:
            zr_new = (
                poly0
                + poly1 * libdevice.pow(zr, 2)
                + poly2 * libdevice.pow(zr, 3)
                + poly3 * libdevice.pow(zr, 4)
            )
            zi_new = zi
        else:
            zr_new = tl.sin(zr)
            zi_new = tl.sin(zi)

        zr = zr_new
        zi = zi_new

        eia_zr = zr * tl.cos(alpha) - zi * tl.sin(alpha)
        eia_zi = zr * tl.sin(alpha) + zi * tl.cos(alpha)

        zr = eia_zr
        zi = eia_zi

        modulus = tl.sqrt(zr * zr + zi * zi)
        modulus_p = libdevice.pow(modulus, p)
        angle = libdevice.atan2(zi, zr) * p

        zr_new = modulus_p * tl.cos(angle)
        zi_new = modulus_p * tl.sin(angle)

        cos_beta = tl.cos(beta)
        sin_beta = tl.sin(beta)
        exp_cr = cr * cos_beta - ci * sin_beta
        exp_ci = cr * sin_beta + ci * cos_beta

        zr = zr_new + exp_cr
        zi = zi_new + exp_ci

        mag_sq = zr * zr + zi * zi

        not_escaped = mag_sq < R * R
        iteration += not_escaped.to(tl.int8)

    iter_float16 = tl.cast(iteration, tl.float16)
    max_float16 = tl.cast(max_iter, tl.float16)
    norm_iter = iter_float16 / max_float16 + 1.0

    escaped = iteration < max_iter

    r = tl.sin(frequency * norm_iter + 0.0)
    g = tl.sin(frequency * norm_iter + 2.0943951)
    b = tl.sin(frequency * norm_iter + 4.1887902)

    r = (r + 1.0) * 0.5
    g = (g + 1.0) * 0.5
    b = (b + 1.0) * 0.5

    r = tl.where(escaped, r, 0.0)
    g = tl.where(escaped, g, 0.0)
    b = tl.where(escaped, b, 0.0)

    r = tl.cast(r * 255, tl.uint8)
    g = tl.cast(g * 255, tl.uint8)
    b = tl.cast(b * 255, tl.uint8)

    channel_stride = H * W
    red_offset = 0 * channel_stride
    green_offset = 1 * channel_stride
    blue_offset = 2 * channel_stride

    tl.store(output_ptr + red_offset + offsets, r, mask=mask)
    tl.store(output_ptr + green_offset + offsets, g, mask=mask)
    tl.store(output_ptr + blue_offset + offsets, b, mask=mask)


class FractalDataset:
    def __init__(self, H, W, max_iter=30, R=4.0, device="cuda"):
        self.H = H
        self.W = W
        self.max_iter = max_iter
        self.R = R
        self.device = device

    @torch.no_grad()
    def batch(
        self,
        list_of_seed,
        polycoeffs=None,
        alpha=None,
        beta=None,
        p=None,
        fractal_type=0,
    ):
        B = len(list_of_seed)
        H, W = self.H, self.W
        device = self.device

        x = torch.linspace(-1.0, 1.0, W)
        y = torch.linspace(-1.0, 1.0, H)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        cr = xx.flatten()
        ci = yy.flatten()

        cr = cr.contiguous()
        ci = ci.contiguous()

        zr = torch.zeros((B, H * W))
        zi = torch.zeros((B, H * W))
        output = torch.empty((B, 3, H * W), dtype=torch.uint8)

        gen = torch.Generator(device="cpu").manual_seed(sum(list_of_seed))

        if alpha is None:
            alpha = torch.rand(B, generator=gen) * 2 * torch.pi
        elif isinstance(alpha, float):
            alpha = torch.ones(B) * alpha
        else:
            alpha = torch.tensor(alpha)

        if beta is None:
            beta = torch.rand(B, generator=gen) * 3.0
        elif isinstance(beta, float):
            beta = torch.ones(B) * beta
        else:
            beta = torch.tensor(beta)

        if p is None:
            p = torch.rand(B, generator=gen) * 5.0
        elif isinstance(p, float):
            p = torch.ones(B) * p
        else:
            p = torch.tensor(p)

        if polycoeffs is None:
            poly0 = torch.rand(B, generator=gen)
            poly1 = torch.rand(B, generator=gen)
            poly2 = torch.rand(B, generator=gen)
            poly3 = torch.rand(B, generator=gen)
        else:
            polycoeffs = torch.tensor(polycoeffs)
            poly0 = torch.ones(B) * polycoeffs[0]
            poly1 = torch.ones(B) * polycoeffs[1]
            poly2 = torch.ones(B) * polycoeffs[2]
            poly3 = torch.ones(B) * polycoeffs[3]

        frequency = torch.rand(B, generator=gen) * 10.0

        alpha = alpha.to(device)
        beta = beta.to(device)
        p = p.to(device)
        poly0 = poly0.to(device)
        poly1 = poly1.to(device)
        poly2 = poly2.to(device)
        poly3 = poly3.to(device)
        cr = cr.to(device)
        ci = ci.to(device)
        zr = zr.to(device)
        zi = zi.to(device)
        output = output.to(device)
        frequency = frequency.to(device)

        BLOCK_SIZE = 1024
        num_blocks = (H * W + BLOCK_SIZE - 1) // BLOCK_SIZE

        grid = (B, num_blocks)

        fractal_kernel[grid](
            zr_ptr=zr,
            zi_ptr=zi,
            cr_ptr=cr,
            ci_ptr=ci,
            output_ptr=output.reshape(-1),
            alpha_ptr=alpha,
            beta_ptr=beta,
            p_ptr=p,
            poly0_ptr=poly0,
            poly1_ptr=poly1,
            poly2_ptr=poly2,
            poly3_ptr=poly3,
            frequency_ptr=frequency,
            R=self.R,
            max_iter=self.max_iter,
            H=H,
            W=W,
            fractal_type=fractal_type,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )

        output = output.view(B, 3, H, W)
        return output

    def __len__(self):
        return 2**32


class FractalImageDataset:
    def __init__(
        self,
        num_classes,
        num_samples_per_class,
        image_size,
        max_iter,
        R,
        device,
        train=True,
        seed=0,
    ):
        self.num_classes = num_classes
        self.num_samples_per_class = num_samples_per_class
        self.image_size = image_size
        self.max_iter = max_iter
        self.R = R
        self.device = device
        self.train = train

        self.total_samples = num_classes * num_samples_per_class

        self.seed = seed
        self.counter = 0
        self.polycoeffs_list = []
        for class_idx in range(num_classes):
            polycoeffs = self.get_polycoeffs_for_class(class_idx)
            self.polycoeffs_list.append(polycoeffs)

        self.fractal_dataset = FractalDataset(
            H=image_size, W=image_size, max_iter=max_iter, R=R, device=device
        )

        self.reset()

    def __len__(self):
        return self.total_samples

    def reset(self):
        self.counter = 0

    def __getitem__(self, idx):
        class_idx = idx % self.num_classes
        data_idx = idx % self.total_samples

        polycoeffs = self.polycoeffs_list[class_idx]

        if self.train:
            data_idx += self.total_samples

        image = self.fractal_dataset.batch([data_idx], polycoeffs=polycoeffs)
        image = image[0]
        image = image.float() / 255.0

        label = class_idx

        return image, label

    def get_batch(self, batch_size):
        images, labels = zip(*[self[i + self.counter] for i in range(batch_size)])
        self.counter += batch_size

        return torch.stack(images), torch.tensor(labels).long()

    def get_polycoeffs_for_class(self, class_idx):
        gen = torch.Generator(device="cpu").manual_seed(class_idx)
        polycoeffs = torch.rand(4, generator=gen)
        return polycoeffs


def test_fractal_dataset():
    dataset = FractalImageDataset(
        num_classes=10,
        num_samples_per_class=10,
        image_size=512,
        max_iter=30,
        R=4.0,
        device="cuda",
    )
    images, labels = dataset.get_batch(100)
    print(images.shape)
    print(labels.shape)

    from PIL import Image

    images = images * 255
    images = images.to(torch.uint8)

    for i in range(images.shape[0]):
        image = images[i]
        label = labels[i]
        image = image.permute(1, 2, 0)
        image = image.cpu().numpy()
        image = Image.fromarray(image)
        image.save(f"./content/image_lab_{label}_idx_{i}.png")


def test_fractal_seeding_behavior():
    dataset = FractalImageDataset(
        num_classes=10,
        num_samples_per_class=10,
        image_size=512,
        max_iter=30,
        R=4.0,
        device="cuda",
        seed=0,
    )

    init_image, init_label = dataset[0]

    for i in range(10):
        images, labels = dataset.get_batch(10)
        print(images.shape)
        print(labels)

    # after 10 samples, the counter should be 100
    assert dataset.counter == 100


if __name__ == "__main__":
    test_fractal_dataset()
    test_fractal_seeding_behavior()
