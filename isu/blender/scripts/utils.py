import math
import bpy

def deg2rad(deg):
    return deg * math.pi / 180.0
def float2rad(f):
    # f in the range of 0 and 1, convert to radians
    return f * math.pi 

def set_tree_visible(root, visible: bool):
    if root is None:
        return
    stack = [root]
    while stack:
        obj = stack.pop()
        obj.hide_render = not visible
        stack.extend(obj.children)

def ensure_link(links, from_socket, to_socket):
    # remove any existing link into the target socket, then link
    for l in list(links):
        if l.to_socket == to_socket:
            links.remove(l)
    links.new(from_socket, to_socket)

def recolor_red_to(material_name: str, suitcase_color: str, target_rgba, hue_width=0.10):
    mat = bpy.data.materials[material_name]
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    principled = next(n for n in nodes if n.type == 'BSDF_PRINCIPLED')

    # --- find the current source feeding Base Color ---
    base_in = principled.inputs["Base Color"]
    src_link = next(l for l in links if l.to_node==principled and l.to_socket==base_in)
    src_socket = src_link.from_socket
    links.remove(src_link)

    # --- build a RED mask (0..1) ---
    sep = nodes.new("ShaderNodeSeparateColor"); sep.mode = 'HSV'
    ensure_link(links, src_socket, sep.inputs["Color"])

    sub = nodes.new("ShaderNodeMath"); sub.operation = 'SUBTRACT'; sub.inputs[1].default_value = 0.0
    links.new(sep.outputs["Red"], sub.inputs[0])   # H channel
    abs1 = nodes.new("ShaderNodeMath"); abs1.operation = 'ABSOLUTE'; links.new(sub.outputs[0], abs1.inputs[0])
    one_minus = nodes.new("ShaderNodeMath"); one_minus.operation='SUBTRACT'; one_minus.inputs[0].default_value=1.0
    links.new(sep.outputs["Red"], one_minus.inputs[1])
    abs2 = nodes.new("ShaderNodeMath"); abs2.operation='ABSOLUTE'; links.new(one_minus.outputs[0], abs2.inputs[0])
    mn = nodes.new("ShaderNodeMath"); mn.operation='MINIMUM'
    links.new(abs1.outputs[0], mn.inputs[0]); links.new(abs2.outputs[0], mn.inputs[1])
    rng = nodes.new("ShaderNodeMapRange"); rng.clamp=True
    rng.inputs[1].default_value = 0.0           # From Min
    rng.inputs[2].default_value = hue_width     # From Max (widen if some reds missed)
    rng.inputs[3].default_value = 1.0           # To Min
    rng.inputs[4].default_value = 0.0           # To Max
    links.new(mn.outputs[0], rng.inputs[0])
    mask_socket = rng.outputs["Result"]         # <-- use THIS for Factor everywhere

    # --- create a single Mix node, then set blend per color ---
    try:
        mix = nodes.new("ShaderNodeMix"); mix.data_type = 'RGBA'
        fac, A, B, OUT = "Factor", "A", "B", "Result"
    except RuntimeError:
        mix = nodes.new("ShaderNodeMixRGB")
        fac, A, B, OUT = "Fac", "Color1", "Color2", "Color"
    mix.location = (principled.location.x - 220, principled.location.y)

    # target color node
    rgb = nodes.new("ShaderNodeRGB")

    col = suitcase_color.upper()
    if col == "ANTRACITE":
        mix.blend_type = "MIX"
        rgb.outputs[0].default_value = target_rgba
        principled.inputs["Metallic"].default_value = 0.5
        principled.inputs["Roughness"].default_value = 0.4
    else:
        mix.blend_type = "HUE"
        rgb.outputs[0].default_value = target_rgba
        principled.inputs["Metallic"].default_value = 0.1
        principled.inputs["Roughness"].default_value = 0.8

    # wire graph: original -> A, target -> B, MASK -> Factor, out -> Base Color
    ensure_link(links, src_socket,            mix.inputs[A])
    ensure_link(links, rgb.outputs["Color"],  mix.inputs[B])
    ensure_link(links, mask_socket,           mix.inputs[fac])    # <-- MASK ENFORCED
    ensure_link(links, mix.outputs[OUT],      base_in)
