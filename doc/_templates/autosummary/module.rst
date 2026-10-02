{{ fullname | escape | underline }}

{% if modules %}
.. automodule:: {{ fullname }}

.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
{%- if not item.split('.')[-1].startswith('_') %}
   {{ item }}
{%- endif %}
{%- endfor %}
{% else %}
.. automodule:: {{ fullname }}
   :members:
   :show-inheritance:
{% endif %}
