---
# create lunr store 
---
var store = [ 
    {% for item in site.html_pages %} 
    { 
        "url": {{ item.url | relative_url | jsonify }},
        "title": {{ item.title | jsonify }},
        "topics": {{ item.topics | jsonify }},
        "text": {{ item.content | strip_html | normalize_whitespace | jsonify }}
    }{%- unless forloop.last -%},{%- endunless -%}
    {%- endfor -%}
];
