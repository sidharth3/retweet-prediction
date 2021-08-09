import fetch from 'node-fetch'
import { printFooter, printLogo } from './utils/print'
import { getControlla } from './utils/transforms'
import { hideMessage } from './utils/misc'

export async function init (path, hide = hideMessage()) {
  if (hide) {
    return
  }

  global.fetch = global.fetch || fetch
  const controlla = await getControlla(path)

  printLogo(controlla.logo)
  printFooter(controlla)
}
