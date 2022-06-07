import os

import bpy

from utils.io import ls


class ModelsCollection:
    """ Collection of 3D models in a Blender scene.

    Attributes:
        name (str): Name of the collection.
        scene (bpy.types.Scene): Scene that contains the collection.
    """

    def __init__(self, name, scene):
        """ Initialize the collection.

        Args:
            name (str): Name of the collection.
            scene (bpy.types.Scene): Scene that contains the collection.
        """
        self.name = name
        self.scene = scene

        try:
            self._collection = bpy.data.collections[name]
        except KeyError:
            self._collection = bpy.data.collections.new(name)
            self.scene.collection.children.link(self._collection)

    def get(self, name):
        """ Get a model by name.

        Args:
            name (str): Name of the model.

        Returns:
            bpy.types.Object: The model.
        """
        return self._collection.objects[name]

    def clear(self):
        """ Remove all models from the collection. """
        meshes = set()

        # Remove objects from the collection
        for obj in [o for o in self._collection.objects if o.type == 'MESH']:
            meshes.add(obj.data)
            bpy.data.objects.remove(obj)

        # Also remove meshes that are orphaned after object removal
        for mesh in [m for m in meshes if m.users == 0]:
            bpy.data.meshes.remove(mesh)

    def list(self):
        """ List all models in the collection.

        Returns:
            list: All models in the collection.
        """
        return [o for o in self._collection.objects if o.type == 'MESH']

    def hide_all(self):
        """ Disable rendering of all models in the collection. """
        for model in self.list():
            model.hide_render = True

    def show(self, name):
        """ Enable rendering of the specified model.

        Args:
            name (str): Name of the model to enable rendering for.
        """
        self._collection.objects[name].hide_render = False

    def import_model(self, path, keep_materials=False, clear_scene=True):
        """ Import a model from an OBJ file.

        If the model has multiple meshes, all of them will be merged into a single mesh.
        The model will be named after the file name.

        Args:
            path (str): Path to the OBJ file.
            keep_materials (bool): Whether to keep the materials of the model. If
                                   False, all materials are removed. Default: False.
            clear_scene (bool): Whether to clear the scene before importing the model.
                                If True, existing models are deleted. Default: True.
        """
        if clear_scene:
            self.clear()

        # Import the model and merge all meshes into a single mesh
        bpy.ops.import_scene.obj(filepath=path)
        model = bpy.context.selected_objects[0]
        bpy.context.view_layer.objects.active = model
        bpy.ops.object.join()

        # Rename the model to the file name
        model.name = path.split('/')[-1].split('.')[0]

        # Remove all materials if requested
        if not keep_materials:
            model.data.materials.clear()

        # Move it to the models collection
        c_old = bpy.context.object.users_collection[0]
        c_new = self._collection
        c_new.objects.link(model)
        c_old.objects.unlink(model)

        return model.name

    def import_models(self, path, keep_materials=False, clear_scene=True):
        """ Import all models from OBJ files in a folder.

        Args:
            path (str): Path to the folder containing the OBJ files.
            keep_materials (bool): Whether to keep the materials of the models. If
                                   False, all materials are removed. Default: False.
            clear_scene (bool): Whether to clear the scene before importing the models.
                                If True, existing models are deleted. Default: True.
        """
        if clear_scene:
            self.clear()

        # Import all models in the folder
        for file in ls(path, '.obj'):
            file_path = os.path.join(path, file)
            self.import_model(file_path, keep_materials=keep_materials, clear_scene=False)
