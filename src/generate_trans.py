import os

import bpy
import cv2
import numpy as np


def use_cycles_with_cuda():
    """ Use the CYCLES render engine with GPU rendering enabled. """
    bpy.context.scene.render.engine = 'CYCLES'

    # Enable GPU rendering with CUDA
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.context.scene.cycles.device = 'GPU'

    # Let Blender detects GPU device
    bpy.context.preferences.addons['cycles'].preferences.get_devices()

    print(bpy.context.preferences.addons['cycles'].preferences.compute_device_type)
    for d in bpy.context.preferences.addons['cycles'].preferences.devices:
        d['use'] = 1  # Using all devices, include GPU and CPU
        print(d['name'], d['use'])


def exclude_objects(objects):
    """ Exclude objects from render.

    Args:
        objects (list of str): Names of objects to exclude.
    """
    for o in objects:
        bpy.data.objects[o].hide_render = True


class Lighting:
    """ Lighting conditions in a Blender scene. """

    def __init__(self):
        self.lights = {
            'Ll': bpy.data.objects['L_l'],
            'Lr': bpy.data.objects['L_r'],
        }

    def disable_light(self, name):
        """ Disable a specific light """
        self.lights[name].hide_render = True

    def enable_light(self, name):
        """ Enable a specific light """
        self.lights[name].hide_render = False

    def disable(self):
        """ Disable all lights """
        for name in self.lights.keys():
            self.disable_light(name)

    def enable(self):
        """ Enable all lights """
        for k in self.lights.keys():
            self.lights[k].hide_render = False


class Renderer:
    """ Defines a Blender renderer. """

    def __init__(self, engine, model_name, save_path, hdri_path):
        self.scene = bpy.context.scene
        self.scene.render.engine = engine
        use_cycles_with_cuda()

        self.model_name = '_'.join(model_name.split(',')).lower()
        self.models = model_name.split(',')
        self.orbit = bpy.data.objects[self.models[0]]

        self.save_path = save_path

        self.lighting = Lighting()
        self.lighting.disable()
        self.cameras = {
            'front': bpy.data.objects['C_90'],
            'down': bpy.data.objects['C_45'],
            'up': bpy.data.objects['C_135'],
        }

        # Get list of available HDRIs
        self.hdris = sorted(
            list(filter(lambda x: x.endswith('.hdr') and not x.startswith('._'), os.listdir(hdri_path))))

        # Load all HDRIs
        for hdri in self.hdris:
            bpy.data.images.load(os.path.join(hdri_path, hdri))

    def create_sequence_name(self, perspective):
        """ Create a sequence name.

        Args:
            perspective (str): 'down', 'front', 'up'
            light (str): 'Ll', 'Lr', 'La'
        """
        return f'{self.model_name}/{perspective}'

    def prepare(self, model):
        """ Prepare model for rendering.

        Centers the model at origin, and rescales it to fit inside the
        camera viewport.

        Args:
            model (bpy.types.Object): The model to prepare.
        """
        # Rescale to fit camera frame
        dims = model.dimensions
        dims /= max(dims)

        # Position at origin
        model.select_set(True)
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
        model.location *= 0

    def set_hdri(self, hdri):
        """ Set the HDRI of the world environment.

        Uses a 360 degree HDR image to create a world environment with realistic lighting and textures in the background of the 3D scene.
        """
        world = bpy.data.worlds['World']
        background = world.node_tree.nodes['Environment Texture']
        background.image = bpy.data.images[hdri]

    def render_frame(self, idx, output_path):
        """ Render a single frame.

        Args:
            idx (str): index of the frame to render
            output_path (str): path to save the rendered frame
        """
        image_path = f'{output_path}/images'
        depth_path = f'{output_path}/depth_maps'
        norms_path = f'{output_path}/normals'

        os.makedirs(image_path, exist_ok=True)
        os.makedirs(depth_path, exist_ok=True)
        os.makedirs(norms_path, exist_ok=True)

        # For each environment HDRI
        for hdri in self.hdris:
            self.set_hdri(hdri)  # Activate current HDRI

            # Render scene with original materials and save RGB image
            wid = hdri.split('.')[0].lower()
            bpy.context.scene.render.filepath = (f'{image_path}/{idx}_{wid}.png')
            bpy.ops.render.render(False, animation=False, write_still=True)

        # Save name of the material of each model, and make it 'opaque'
        materials = {}
        for model_name in self.models:
            model = bpy.data.objects[model_name]
            materials[model_name] = model.data.materials[0]
            model.data.materials[0] = bpy.data.materials['white']

        self.lighting.disable()
        # Render scene again with opaque models (to get accurate depth values)
        bpy.ops.render.render(False, animation=False, write_still=False)
        self.lighting.enable()

        z = bpy.data.images['Viewer Node']
        w, h = z.size
        data = np.array(z.pixels[:], dtype=np.float32)
        data = np.reshape(data, (h, w, 4))
        data = np.rot90(data, k=2)
        data = np.fliplr(data)

        # Read the z-buffer data
        depth = data[:, :, 3]

        # Read the surface normals and convert them to unit vectors
        norms = data[:, :, :3]
        length = np.linalg.norm(norms.astype(np.float32), axis=2, keepdims=True)
        length[length == 0] = 1.0
        norms[:, :, :] /= length
        norms[depth >= 1.0, :] = 1.0

        # Save the depth and normal maps
        np.savez_compressed(f'{output_path}/{idx}.npz',
                            dmap=depth.astype(np.float16),
                            nmap=norms.astype(np.float16))

        # Also save the depth and normal maps as PNGs
        cv2.imwrite(os.path.join(depth_path, '{}.png'.format(idx)), depth * 255)

        norms = (norms + 1) / 2
        norms[depth >= 1.0, :] = 1.0
        cv2.imwrite(os.path.join(norms_path, '{}.png'.format(idx)), norms[:, :, ::-1] * 255)

        # Restore materials of all models to original
        for model_name in self.models:
            model = bpy.data.objects[model_name]
            model.data.materials[0] = materials[model_name]

    def render_sequence(self, sequence):
        """ Render a sequence of images for a given object.

        Args:
            sequence (str): name of the sequence
        """
        # Create sequence output path (if needed).
        sequence_dir = f'{self.save_path}/{sequence}/'
        os.makedirs(sequence_dir, exist_ok=True)

        for i in range(0, 360):
            self.orbit.rotation_euler[2] = np.radians(i)  # rotate around z-axis
            self.render_frame(idx=f'{str(i).zfill(4)}', output_path=sequence_dir)

    def render_sequences_from_direction(self, direction):
        """ Render all sequences from a given camera direction.

        Args:
            direction (str): the camera direction to use for renders
        """
        # Render the model with all lights on
        self.lighting.enable()
        self.render_sequence(sequence=self.create_sequence_name(direction))

    def render_sequences(self):
        """ Render all sequences for all cameras. """
        for direction in self.cameras.keys():
            bpy.context.scene.camera = self.cameras[direction]  # Activate current camera
            self.render_sequences_from_direction(direction)


def parse_args():
    import sys  # to get command line args
    import argparse  # to parse options for us and print a nice help message

    # get the args passed to blender after "--", all of which are ignored by
    # blender so scripts may receive their own arguments
    argv = sys.argv

    if "--" not in argv:
        argv = []  # as if no args are passed
    else:
        argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str)
    parser.add_argument('hdri_path', type=str, default='../data/hdris/')
    parser.add_argument('--save_path', '-s', type=str, default='../tmp/')
    parser.add_argument('--engine', type=str, default='CYCLES')
    return parser.parse_args(argv)


def main(args):
    """ Render all possible sequences for a given model. """
    render = Renderer(args.engine, args.model_name, args.save_path, args.hdri_path)
    render.render_sequences()


if __name__ == '__main__':
    main(parse_args())
