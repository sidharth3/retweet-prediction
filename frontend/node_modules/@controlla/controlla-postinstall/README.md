# Controlla postinstall

Lightweight npm postinstall message

## Installation

```
npm install --save @controlla/postinstall
```

And in your `package.json` add:

```json
{
  ...
  "scripts": {
    "postinstall": "controlla-postinstall || exit 0"
  },
  "controlla": {
    "url": "https://controlla.github.io/vuemerang/"
  }
  ...
}
```

## Configuration

Configuration is applied through your project's `package.json`.

A full configuration looks like:

```json
{
  "controlla": {
    "url": "https://controlla.github.io/vuemerang/",
    "logoUrl": "https://controlla.github.io/vuemerang/logo.txt"
  }
}
```

---

Note: This is a lightweight alternative to the [controlla-cli](https://github.com/controlla/controlla-cli) that offers a more complete postinstall message.
