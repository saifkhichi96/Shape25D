""" Generates synthetic RGB-D data using Blender. """
import os
import sys

import bpy
import cv2
import numpy as np

root = os.path.dirname(bpy.data.filepath)
if root not in sys.path:
    sys.path.append(root)

from models import ModelsCollection


class SceneData:
    """ Data in the Blender scene.

    Attributes:
        scene (bpy.types.Scene): The scene.
        models (ModelsCollection): The collection of models to render.
        cameras (bpy.types.Collection): The collection of cameras in the scene.
        lights (bpy.types.Collection): The collection of lights in the scene.
    """

    def __init__(self, name, resolution):
        """ Initialize the scene data.

        Args:
            name (str): The name of the scene.
            resolution (tuple): The resolution of the scene as (width, height).
        """
        bpy.ops.scene.new(type='NEW')
        self.scene = bpy.context.scene
        self.scene.render.resolution_x = resolution[0]
        self.scene.render.resolution_y = resolution[1]
        self.scene.name = name

        self.models = ModelsCollection('models', self.scene)

        self.cameras = bpy.data.collections.new('cameras')
        self.scene.collection.children.link(self.cameras)

        self.lights = bpy.data.collections.new('lights')
        self.scene.collection.children.link(self.lights)

        # Enabled Combined, Z and Normal render passes
        bpy.context.view_layer.use_pass_combined = True
        bpy.context.view_layer.use_pass_z = True
        bpy.context.view_layer.use_pass_normal = True

        # Enable compositor nodes and create a viewer node
        self.scene.use_nodes = True
        tree = self.scene.node_tree
        tree.nodes.new('CompositorNodeViewer')
        tree.nodes.new('CompositorNodeNormalize')

        tree.links.new(tree.nodes['Render Layers'].outputs['Normal'], tree.nodes['Viewer'].inputs['Image'])
        tree.links.new(tree.nodes['Render Layers'].outputs['Depth'], tree.nodes['Normalize'].inputs[0])
        tree.links.new(tree.nodes['Normalize'].outputs[0], tree.nodes['Viewer'].inputs['Alpha'])

    def add_camera(self, name, position, rotation):
        """ Add a camera to the scene.

        Args:
            name (str): The name of the camera.
            position (tuple): The x, y, z position of the camera.
            rotation (tuple): The euler rotation of the camera in degrees.
        """
        camera_data = bpy.data.cameras.new(name=name)
        camera = bpy.data.objects.new(name, object_data=camera_data)
        camera.location = position
        camera.rotation_euler = tuple([x * np.pi / 180.0 for x in rotation])
        self.cameras.objects.link(camera)

    def add_light(self, name, position, rotation, energy, color=(1.0, 1.0, 1.0), type='SUN', spot_size=25.0):
        """ Add a light to the scene.

        Args:
            name (str): The name of the light.
            position (tuple): The x, y, z position of the light.
            rotation (tuple): The euler rotation of the light in degrees.
            energy (float): The intensity of the light in Watts. In case of sun, this
                            is the strength of sunlight in Watts per square meter.
            color (tuple): The RGB color of the light. Default: (1.0, 1.0, 1.0).
            type (str): The type of light. 'SUN', 'POINT', 'SPOT', or 'AREA'.
            spot_size (float): The size of the spot light in degrees. Only used if
                               type is 'SPOT'.
        """
        light_data = bpy.data.lights.new(name=name, type=type)
        light_data.energy = energy
        light_data.color = color
        if type == 'SPOT':
            light_data.spot_size = spot_size

        light = bpy.data.objects.new(name, object_data=light_data)
        light.location = position
        light.rotation_euler = tuple([x * np.pi / 180.0 for x in rotation])
        self.lights.objects.link(light)


class Renderer:
    """ The scene renderer.

    Attributes:
        save_path (str): The path to save the rendered images to.
        scene_data (SceneData): The scene data.
        engine (str): The rendering engine to use. Either 'CYCLES' or 'BLENDER_EEVEE'.
        device (str): The device to use. Either 'CPU' or 'CUDA'. Currently CUDA is only
                      supported by the Cycles engine.
    """

    def __init__(self, save_path, scene_data, engine='BLENDER_EEVEE', device='CPU'):
        """ Initialize the renderer.

        Args:
            save_path (str): The path to save the rendered images to.
            scene_data (SceneData): The scene data.
            engine (str): Rendering engine to use. Default: 'BLENDER_EEVEE'. Can
                          also be 'CYCLES'
            device (str): Device to use for rendering. Can be 'CPU' or 'CUDA'.
                          Default: 'CPU'. Only used if engine is 'CYCLES'.

        """
        self.save_path = save_path
        self.scene_data = scene_data
        self.engine = engine
        self.device = device
        self.init_device_and_engine()

    def init_device_and_engine(self):
        """ Initialize the device and the rendering engine. """
        self.scene_data.scene.render.engine = self.engine

        # If using Cycles and CUDA requested, try to enable GPU rendering
        if self.device == 'CUDA' and self.engine == 'CYCLES':
            try:
                bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
                bpy.context.scene.cycles.device = 'GPU'
                bpy.context.preferences.addons['cycles'].preferences.get_devices()
                for d in bpy.context.preferences.addons['cycles'].preferences.devices:
                    d['use'] = 1  # Using all devices, include GPU and CPU
            except Exception as e:
                print(e)
                print('Failed to enable CUDA. Falling back to CPU.')
                bpy.context.scene.cycles.device = 'CPU'

    def _render_frame(self, idx, subdir=''):
        """ Render the scene and save the output to disk.

        Args:
            idx (str): The index of the image to render.
            subdir (str): The subdirectory to save the data to. If empty, the
                          data will be saved to the root of the save path. If
                          not empty, the data will be saved to the subdirectory
                          with the given name. This is useful for saving data
                          to different subdirectories for different scenarios.
                          Default is empty.
        """
        output_path = os.path.join(self.save_path, subdir)
        os.makedirs(output_path, exist_ok=True)

        # Render the scene and save RGB image
        self.scene_data.scene.render.filepath = os.path.join(output_path, 'rgb_{}.png'.format(idx))
        bpy.ops.render.render(write_still=True)

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

        # Save depth and normal maps as numpy arrays
        np.savez_compressed(os.path.join(output_path, 'd_{}.npz'.format(idx)),
                            dmap=depth.astype(np.float32),
                            nmap=norms.astype(np.float32))

        # # Also save the depth and normal maps as PNGs
        # cv2.imwrite(os.path.join(output_path, '{}_depth.png'.format(idx)), depth * 255)
        #
        # norms = (norms + 1) / 2
        # norms[depth >= 1.0, :] = 1.0
        # cv2.imwrite(os.path.join(output_path, '{}_normals.png'.format(idx)), norms[:, :, ::-1] * 255)

    def _render_sequence(self, model_name, camera, light, render_angles):
        """ Render a sequence of images.

        Args:
            model_name (str): The name of the model to render.
            camera (str) : The name of the camera to render from.
            light (str): The name of the light to render from.
            render_angles (range): Range of angles to render. The model is rotated
                                   around the z-axis by specified angle in each
                                   frame.
        """
        sequence = f'{model_name.lower()}/{light}_{camera}'
        model = self.scene_data.models.get(name=model_name)
        for i in render_angles:
            model.rotation_euler[2] = np.radians(i)  # rotate the model around z-axis
            self._render_frame(idx=f'{str(i).zfill(4)}', subdir=sequence)

    def render(self, model_name, render_angles, always_on=None, exclude=None):
        """ Render the scene with all cameras under all lighting setups.

        Only renders the model with the given name. If there are multiple models
        in scene_data, the other models will be hidden.

        Args:
            model_name (str): The name of the model to render.
            render_angles (range): Range of angles to render. The model is rotated
                                   around the z-axis by specified angle in each
                                   frame.
            always_on (list): The names of the lights to always keep on. If None,
                              each light is turned on one at a time. If not None,
                              in addition to turning on each light, the lights in
                              always_on are turned on in every frame.
            exclude (list): The names of the lights to exclude from the rendering
                            individually. If None, all lights are rendered.
        """
        # Hide all models except the one to render
        self.scene_data.models.hide_all()
        self.scene_data.models.show(name=model_name)

        # Render scene for each camera
        for camera in self.scene_data.cameras.objects:
            print('Using camera: {}'.format(camera.name))
            bpy.context.scene.camera = camera  # set active camera

            # Disable all lights (except the ones in always_on)
            for light in self.scene_data.lights.objects:
                if always_on is not None and light.name in always_on:
                    light.hide_render = False
                else:
                    light.hide_render = True

            # For each light, enable it and render scene
            for light in self.scene_data.lights.objects:
                if exclude is not None and light.name in exclude:
                    continue

                light.hide_render = False
                self._render_sequence(model_name, camera.name, light.name, render_angles)

                # Turn off light again (if it is not in always_on)
                if always_on is None or light.name not in always_on:
                    light.hide_render = True

            # Turn on all lights
            for light in self.scene_data.lights.objects:
                light.hide_render = False

            # Render the scene again with all lights on
            self._render_sequence(model_name, camera.name, 'La', render_angles)


class TLessGenerator:
    """ Class to generate texture-less data.

    Attributes:
        scene_data (SceneData): The scene data object.
        renderer (Renderer): The renderer object.
    """

    def __init__(self, save_path, resolution=(512, 512),
                 engine='BLENDER_EEVEE', device='CPU'):
        """ Initialize the generator.

        Args:
            save_path (str): Path to save the generated data.
            resolution (tuple): Resolution of the rendered images.
            engine (str): Rendering engine to use. Default: 'BLENDER_EEVEE'. Can
                          also be 'CYCLES'
            device (str): Device to use for rendering. Can be 'CPU' or 'CUDA'.
                          Default: 'CPU'. Only used if engine is 'CYCLES'.
        """
        self.scene_data = SceneData(name='Default', resolution=resolution)

        # Create three cameras looking at origin from different directions
        self.scene_data.add_camera('down', position=(0, -1.4, 1.4), rotation=(45, 0, 0))
        self.scene_data.add_camera('front', position=(0, -2, 0), rotation=(90, 0, 0))
        self.scene_data.add_camera('up', position=(0, -1.4, -1.4), rotation=(135, 0, 0))

        # Add two halogen lamps on front-left and front-right of origin
        self.scene_data.add_light('Ll', position=(3, -3, 0), rotation=(90, 0, 45),
                                  energy=1000, color=(1.0, 0.945, 0.875),
                                  type='SPOT', spot_size=25)
        self.scene_data.add_light('Lr', position=(-3, -3, 0), rotation=(90, 0, -45),
                                  energy=1000, color=(1.0, 0.945, 0.875),
                                  type='SPOT', spot_size=25)

        # Add sunlight above origin
        self.scene_data.add_light('Ls', position=(0, 0, 5), rotation=(0, 0, 0),
                                  energy=2.0, color=(0.785, 0.883, 1.0), type='SUN')

        # Add an ambient point light below origin to illuminate the bottom faces of models
        self.scene_data.add_light('ambient', position=(0, 0, -5), rotation=(0, 0, 0),
                                  energy=100, color=(1.0, 1.0, 1.0), type='POINT')

        self.renderer = Renderer(save_path, self.scene_data, engine=engine, device=device)

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

    def generate(self, models_path, render_angles):
        """ Generate the data.

        Args:
            models_path (str): Path to folder containing OBJ models.
            render_angles (range): Range of angles to render. The model is rotated
                                 around the z-axis by specified angle in each
                                 frame.
        """
        # Load models
        print('Loading models...')
        self.scene_data.models.import_models(models_path, keep_materials=False, clear_scene=True)
        print('Found {} models.'.format(len(self.scene_data.models.list())))

        # Render each model individually
        for model in self.scene_data.models.list():
            print('Rendering model {}...'.format(model.name))
            self.prepare(model)
            self.renderer.render(model.name, render_angles,
                                 always_on=['Ls', 'ambient'], exclude=['ambient'])


def parse_args():
    """ Parse command line arguments. """
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
    parser.add_argument('models_path', type=str, help='Path of folder containing OBJ files to render.')
    parser.add_argument('--save_path', '-s', type=str, default='~/tmp/renderings')
    parser.add_argument('--engine', '-e', type=str, default='BLENDER_EEVEE',
                        help='The renderer to use. BLENDER_EEVEE and CYCLES available. CYCLES cannot be used in '
                             'headless mode. Default: BLENDER_EEVEE')
    parser.add_argument('--use_gpu', '-g', action='store_true',
                        help='Use GPU for faster rendering (if available). It can only be used with CYCLES renderer.')
    return parser.parse_args(argv)


def main(args):
    gen = TLessGenerator(args.save_path, engine=args.engine, device='CPU' if not args.use_gpu else 'CUDA')
    gen.generate(args.models_path, range(1, 2))


if __name__ == '__main__':
    main(parse_args())
