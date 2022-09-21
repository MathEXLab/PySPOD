---
layout: default
---

Text can be **bold**, _italic_, or ~~strikethrough~~.

[Link to another page](./another-page.html).

There should be whitespace between paragraphs.

# Header 1

This is a normal paragraph following a header. GitHub is a code hosting platform for version control and collaboration. It lets you and others work together on projects from anywhere.

## Header 2

> This is a blockquote following a header.
>
> When something is important enough, you do it even if the odds are not in your favor.

### Header 3

```js
// Javascript code with syntax highlighting.
var fun = function lang(l) {
  dateformat.i18n = require('./lang/' + l)
  return true;
}
```

```ruby
# Ruby code with syntax highlighting
GitHubPages::Dependencies.gems.each do |gem, version|
  s.add_dependency(gem, "= #{version}")
end
```

```python
# Python code with syntax highlighting
def oblique_projection(Phi_tilde, W_phi, W, Q, svd=True):
	if svd:
		PhiTWPhi = Phi_tilde.conj().T @ (W_phi * Phi_tilde)
		PhiTWQ = Phi_tilde.conj().T @ (W * Q)
		u, l, v = np.linalg.svd(PhiTWPhi)
		l_inv = np.zeros([len(l), len(l)], dtype='complex_')
		for i in range(len(l)):
			if (l[i] > 1e-10):
				l_inv[i,i] = 1 / l[i]
		PhiTWPhi_inv = (v.conj().T @ l_inv) @ u.conj().T
		coeffs = PhiTWPhi_inv @ PhiTWQ
	else:
		PhiTWPhi = Phi_tilde.conj().T @ (W_phi * Phi_tilde)
		PhiTWQ = Phi_tilde.conj().T @ (W * Q)
		tmp1_inv = np.linalg.pinv(PhiTWPhi)
		coeffs = tmp1_inv @ PhiTWQ
	return coeffs
end
```

#### Header 4

*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.

##### Header 5

1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.

###### Header 6

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

### There's a horizontal rule below this.

* * *

### Here is an unordered list:

*   Item foo
*   Item bar
*   Item baz
*   Item zip

### And an ordered list:

1.  Item one
1.  Item two
1.  Item three
1.  Item four

### And a nested list:

- level 1 item
  - level 2 item
  - level 2 item
    - level 3 item
    - level 3 item
- level 1 item
  - level 2 item
  - level 2 item
  - level 2 item
- level 1 item
  - level 2 item
  - level 2 item
- level 1 item

### Small image

![Octocat](https://github.githubassets.com/images/icons/emoji/octocat.png)

### Large image

![Branching](https://guides.github.com/activities/hello-world/branching.png)


### Definition lists can be used with HTML syntax.

<dl>
<dt>Name</dt>
<dd>Godzilla</dd>
<dt>Born</dt>
<dd>1952</dd>
<dt>Birthplace</dt>
<dd>Japan</dd>
<dt>Color</dt>
<dd>Green</dd>
</dl>

```
Long, single-line code blocks should not wrap. They should horizontally scroll if they are too long. This line should be long enough to demonstrate this.
```

```
The final element.
```
