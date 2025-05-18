from django import template

register = template.Library()

@register.filter
def get_item(nested_list, index):
    try:
        return nested_list[index]
    except (IndexError, TypeError):
        return -1

@register.filter
def grid_range(size):
    try:
        return range(size)
    except TypeError:
        return range(0)