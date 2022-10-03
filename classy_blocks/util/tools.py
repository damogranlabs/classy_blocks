import importlib.resources

import jinja2

# mesh utils
def template_to_dict(template_name, dict_path, context):
    """ renders template with context to produce an OF dictionary
    (or anything else, depends on the template) """
    template_text = importlib.resources.read_text(__package__, template_name)

    template = jinja2.Template(template_text)
    mesh_file = open(dict_path, "w")
    mesh_file.write(template.render(context))
    mesh_file.close()
