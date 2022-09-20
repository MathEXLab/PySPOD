

# H1
## H2
### H3

**bold text**

*italicized text*

> blockquote

1. First item
2. Second item
3. Third item

- First item
- Second item
- Third item

`code`

---

[title](https://www.example.com)

![alt text](image.jpg)

| Syntax | Description |
| ----------- | ----------- |
| Header | Title |
| Paragraph | Text |

```
{
  "firstName": "John",
  "lastName": "Smith",
  "age": 25
}
```

- [x] Write the press release
- [ ] Update the website
- [ ] Contact the media

I need to highlight these ==very important words==.

~~The world is flat.~~


term
: definition

### My Great Heading {#custom-id}

{% capture source %}GitHub Pages now allows *any* branch's root or "docs" folder to be [selected as the source](https://docs.github.com/en/free-pro-team@latest/github/working-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site). 
Following earlier conventions, creating a branch named `gh-pages` will automatically active it as the source. 

Keep in mind that until recently the default branch was called "master", rather than "main", so older documentation may still use that terminology. 
{% endcapture %}
{% include alert.html text=source color="info" %}
