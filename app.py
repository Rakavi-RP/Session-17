import gradio as gr
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from PIL import Image
import torch.nn.functional as F

# Initialize models and tokenizer
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(torch_device)

vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(torch_device)
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").to(torch_device)

scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

def artistic_enhancement_loss(latents):
    with torch.no_grad():
        scaled_latents = 1 / 0.18215 * latents
        images = vae.decode(scaled_latents).sample

    images = (images + 1) / 2

    grayscale = images.mean(dim=1, keepdim=True)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=images.device).float().view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=images.device).float().view(1, 1, 3, 3)
    edge_x = F.conv2d(grayscale, sobel_x, padding=1)
    edge_y = F.conv2d(grayscale, sobel_y, padding=1)
    edges = torch.sqrt(edge_x ** 2 + edge_y ** 2)

    laplacian_filter = torch.tensor([[[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]]], device=images.device).float()
    laplacian_edges = F.conv2d(grayscale, laplacian_filter, padding=1)
    edges = torch.clamp(edges + laplacian_edges * 0.5, 0, 1)

    kernel_size = 9
    sigma = 4.0
    kernel_range = torch.linspace(-(kernel_size//2), kernel_size//2, kernel_size, device=images.device)
    x_kernel, y_kernel = torch.meshgrid(kernel_range, kernel_range, indexing='ij')
    gaussian_kernel = torch.exp(-(x_kernel**2 + y_kernel**2) / (2 * sigma**2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)

    blurred_channels = []
    for c in range(images.shape[1]):
        blurred_channel = F.conv2d(images[:, c:c+1, :, :], gaussian_kernel, padding=kernel_size//2)
        blurred_channels.append(blurred_channel)
    blurred = torch.cat(blurred_channels, dim=1)

    blur_weight = torch.exp(-edges * 8)
    background_blurred = images * blur_weight + blurred * (1 - blur_weight)

    y, x = torch.meshgrid(torch.linspace(-1, 1, images.shape[2], device=images.device),
                           torch.linspace(-1, 1, images.shape[3], device=images.device))
    distance_from_center = torch.sqrt(x**2 + y**2)
    focus_mask = torch.exp(-distance_from_center * 7).unsqueeze(0).unsqueeze(0)
    deep_focus = images * focus_mask + background_blurred * (1 - focus_mask)

    vignette = 1.0 - torch.clamp(distance_from_center * 0.7, 0, 1)
    vignette = vignette.unsqueeze(0).unsqueeze(0)
    deep_focus = deep_focus * vignette

    cartoonified = deep_focus + edges * 0.3
    cartoonified = torch.clamp(cartoonified, 0, 1)

    loss = torch.pow(images - cartoonified, 2).mean()

    latent_grad = torch.ones_like(latents) * 0.1
    latent_grad *= min(0.7, loss.item() * 1.5)

    return loss * 5.0, latent_grad

def latents_to_pil(latents):
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def generate_image(prompt, style):
    text_input = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents = torch.randn((1, unet.config.in_channels, 64, 64), device=torch_device)
    latents = latents * scheduler.init_noise_sigma

    for t in scheduler.timesteps:
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        with torch.enable_grad():
            latents_for_loss = latents.detach().requires_grad_()
            loss, direct_grad = artistic_enhancement_loss(latents_for_loss)
            latents = latents - 0.9 * direct_grad

        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    image = latents_to_pil(latents)[0]
    return image

def display_examples():
    examples = [
        ("A girl under moonlight", "midjourney", "generated images/Prompt 1.png"),
        ("Kids playing with pets in park", "concept_art", "generated images/Prompt 2.png"),
        ("Birds in a garden", "cosmic_galaxy", "generated images/Prompt 3.png")
    ]
    return examples

with gr.Blocks() as demo:
    gr.Markdown("# Stable Diffusion")
    
    with gr.Row():
        prompt = gr.Textbox(label="Enter your prompt")
        style = gr.Dropdown(choices=["midjourney", "concept_art", "cosmic_galaxy", "fireworks", "moeb_style"], label="Choose a style")
    
    generate_button = gr.Button("Generate Image")
    output_image = gr.Image(label="Generated Image")
    
    generate_button.click(fn=generate_image, inputs=[prompt, style], outputs=output_image)
    
    gr.Markdown("## Example of Generated Images")
    example_images = display_examples()
    for example in example_images:
        gr.Markdown(f"**Prompt**: {example[0]}")
        gr.Image(example[2])

demo.launch()
