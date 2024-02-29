"""
This is completely from some guy on a blender forum.
I just copied it here to extract my data and show how I generated my vertices.csv data.
It does not work with the current version of python outside of blender, so it can only be run in blender.
Source: https://blenderartists.org/t/how-do-i-export-an-objects-vertex-positions/1397962/21
Also obviously I edited it a bit.
"""

import bpy
import os


def write_verts():
    obj = bpy.context.object

    if obj is None or obj.type != "MESH":
        return

    # Output geometry
    obj_eval = obj.evaluated_get(bpy.context.view_layer.depsgraph)
    filepath = "D:/Downloads/uploads_files_1983331_ZeroTwo+Chibi+Model/vertices.csv"

    with open(filepath, "w+") as file:
        # Write the header, pos x | pos y | pos z
        file.write("pos_x,pos_y,pos_z\n")
        file.write("numeric,numeric,numeric\n")

        for v in obj_eval.data.vertices:
            # ":.3f" means to use 3 fixed digits after the decimal point.
            file.write(f",".join(f"{c:.3f}" for c in v.co) + "\n")

    print(f"File was written to {os.path.join(os.getcwd(), filepath)}")


if __name__ == "__main__":
    write_verts()