#!/usr/bin/env node

require('../dist/controlla-postinstall.cjs')
  .init(process.argv.length > 2 ? process.argv[2] : '.')
  .then(() => process.exit(0))
  .catch(() => process.exit(1))
