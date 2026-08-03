import base64
from isu.sim2real.gptimage1 import response_gptimage1

def call_sim2real(image_path_sim: str, dummy: bool, gt: dict) -> str:
    if dummy:
        return image_path_sim
    
    prompt_optimized_with_gt = f"""
    Keep the subject, pose, proportions, lighting, hand positions, and camera angle exactly the same as the first image.
    Do not add, remove, or move any objects in the scene. Preserve the exact composition.

    Transform only the visual style so it appears realistic, using the second image strictly as a reference for photographic quality.
    Do not use clothing from the second image; keep all clothing from the first image unchanged.

    If the driver appears bald, add natural-looking hair aligned with the reference style.

    Scene description:
    A {gt['emotion'].lower()} {gt['gender'].lower()} driver is sitting in the car with open eyes.
    The driver is holding {"a" if gt['phone']=="YES" else "no"} phone in the **right hand**, exactly as in the original image.
    The driver { "is wearing a seatbelt." if gt['safety_belt']=="YES" else "is not wearing a seatbelt." }
    { "No suitcase is present." if gt['suitcase']=="NO" else f"A {gt['suitcase_color'].lower()} suitcase is visible on {gt['suitcase_location'].replace('_', ' ').lower()}." }
    """

    print(f"Sim2Real prompt: {prompt_optimized_with_gt}")

    try:
        response = response_gptimage1(
            prompt=prompt_optimized_with_gt,
            reference_image_path="isu/sim2real/reference.jpg",
            image_path=image_path_sim
        )
    except Exception as e:
        print(f"Sim2Real failed for image: {image_path_sim} with error: {e}")
        return image_path_sim  # Fallback to simulated image
    
    img_data = response.data[0].b64_json
    image_bytes = base64.b64decode(img_data)
    image_path_sim2real = image_path_sim.replace("_sim.png", "_sim2real.png")
    with open(image_path_sim2real, "wb") as img_file:
        img_file.write(image_bytes)
        img_file.close()
    #print(f"Sim2Real succeeded for image: {image_path_sim}, saved to: {image_path_sim2real}")
    return image_path_sim2real