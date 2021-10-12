import jinja2
import numpy as np

# mesh utils
def template_to_dict(template_path, dict_path, context):
    """ renders template with context to product a dictionary (or anything else) """
    template_file = open(template_path, "r")
    template_text = template_file.read()
    template_file.close()

    template = jinja2.Template(template_text)
    mesh_file = open(dict_path, "w")
    mesh_file.write(template.render(context))
    mesh_file.close()

def get_count(length, start_thickness, cell_expansion_ratio, tol=1e-6):
    """ returns the number of cells required to fill 'length' with cells
    of specified start thickness and cell-to-cell expansion ratio """
    if abs(cell_expansion_ratio - 1) > tol:
        c = np.log(1- length/start_thickness*(1-cell_expansion_ratio))/np.log(cell_expansion_ratio)
    else:
        c = length/start_thickness
    
    return int(c) + 1

def get_ratio(count, cell_expansion_ratio):
    if count <= 1:
        raise ValueError("Cell count must be greater than 1")
    
    return cell_expansion_ratio**(count-1)