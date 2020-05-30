import jinja2

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
