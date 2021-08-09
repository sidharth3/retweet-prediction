import { reportAndThrowError, stripLeadingSlash } from './misc'
import { fetchLogo, fetchPkg } from './fetch'

export const controllaSlugFromUrl = url => url.substr(url.lastIndexOf('/') + 1).toLowerCase().replace(/\.json/g, '')

export const controllaUrl = pkg => {
  const url = pkg.controlla && pkg.controlla.url

  if (!url) {
    reportAndThrowError('No controlla URL set!')
  }

  return url
}

// use pkg.controlla.logo for "legacy"/compatibility reasons
export const controllaLogoUrl = pkg => pkg.controlla.logo || pkg.controlla.logoUrl || false

export const controllaDonationText = pkg => (pkg.controlla.donation && pkg.controlla.donation.text) || 'Visit:'

export const getControlla = async pkgPath => {
  const pkg = fetchPkg(pkgPath)
  const url = controllaUrl(pkg)
  const baseControlla = {
    url,
    slug: pkg.name,
    logoUrl: controllaLogoUrl(pkg),
    donationUrl: controllaDonationUrl(pkg),
    donationText: controllaDonationText(pkg)
  }
  const logoUrl = baseControlla.logoUrl

  const logo = await fetchLogo(logoUrl)

  return Object.assign(baseControlla, { logo })
}

export const controllaDonationUrl = pkg => {
  const defaultDonationAmount = pkg.controlla.donation && pkg.controlla.donation.amount

  let donateUrl = `${controllaUrl(pkg)}/`

  if (defaultDonationAmount) {
    return `${donateUrl}${defaultDonationAmount}`
  }

  return donateUrl
}

export const retrieveDonationSlug = pkg => {
  const rawDonationSlug = (pkg.controlla.donation && pkg.controlla.donation.slug)

  if (!rawDonationSlug) {
    return ''
  }

  return stripLeadingSlash(rawDonationSlug)
}
