import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from utils import *

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image_internvl2(image_file, input_size=448, max_num=12):
    '''
    Image loader for InternVL2
    '''
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def generate_answer_llama32(prompt, max_tokens, image_path, evidence_images, model, tokenizer):

    content = [{"type": "text", "text": prompt}]
    content += [{"type": "image"}]
    all_images = [Image.open(image_path)]
    if len(evidence_images)!=0:
        #each evidence image is a tuple with the accompanying text and the image
        for ev in evidence_images:
            if ev[0]=='map':
                content += [{"type": "text", "text": 'A map of the candidate location.'}]
                content += [{"type":"image"}]
                all_images.append(Image.open(ev[1]))
            elif ev[0]=='satellite':
                content += [{"type": "text", "text": 'A satellite image of the candidate location.'}]
                content += [{"type":"image"}]
                all_images.append(Image.open(ev[1]))
            else:
                content += [{"type": "text", "text": ev[0]}]

    messages=[
        {
            "role": "user",
            'content':content
        }]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(
            all_images,
            text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)
    
    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=False
    )

    generated_tokens = output.sequences[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response


def generate_answer_qwen25vl(prompt, max_tokens, image_path, evidence_images, model, tokenizer):
    from qwen_vl_utils import process_vision_info
    content = [{"type": "text", "text": prompt}]
    content += [{"type": "image", "image": image_path}]
    if len(evidence_images)!=0:
        #each evidence image is a tuple with the accompanying text and the image
        for ev in evidence_images:
            if ev[0]=='map':
                content += [{"type": "text", "text": 'A map of the candidate location.'}]
                content += [{"type":"image","image":ev[1]}]
            elif ev[0]=='satellite':
                content += [{"type": "text", "text": 'A satellite image of the candidate location.'}]
                content += [{"type":"image","image":ev[1]}]
            else:
                content += [{"type": "text", "text": ev[0]}]

    messages=[
        {
            "role": "user",
            'content':content
        }]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = tokenizer(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    output_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, temperature=0, top_p=1)
    generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]   
    return response

def generate_answer_internvl2(prompt, max_tokens, image_path, evidence_images, model, tokenizer):

    prompt = '<image>\n' + prompt
    pixel_values1 = load_image_internvl2(image_path, max_num=12).to(torch.bfloat16).cuda()
    if len(evidence_images) > 0:
        for ev in evidence_images:
            if ev[0]=='map':
                prompt += 'The second image shows a map of the candidate location.\n<image>\n'
            elif ev[0]=='satellite':
                prompt += 'The second image shows a map of the candidate location.\n<image>\n'
            else:
                pass

            pixel_values2 = load_image_internvl2(ev[1], max_num=12).to(torch.bfloat16).cuda()
            pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
    else:
        pixel_values = pixel_values1
    generation_config = dict(max_new_tokens=max_tokens, do_sample=False)
    response = model.chat(tokenizer, pixel_values, prompt, generation_config)
    return response
