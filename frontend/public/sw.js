const CACHE_NAME = "fiu-sahayak-cache-v1"
const URLS_TO_CACHE = ["/", "/manifest.json"]

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(URLS_TO_CACHE)
    })
  )
})

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) =>
      Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            return caches.delete(cacheName)
          }
          return undefined
        })
      )
    )
  )
})

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") {
    return
  }

  event.respondWith(
    caches.match(event.request).then((response) => {
      if (response) {
        return response
      }

      return fetch(event.request)
        .then((fetchResponse) => {
          if (
            !fetchResponse ||
            fetchResponse.status !== 200 ||
            fetchResponse.type !== "basic"
          ) {
            return fetchResponse
          }

          const responseToCache = fetchResponse.clone()

          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, responseToCache)
          })

          return fetchResponse
        })
        .catch(() => caches.match("/"))
    })
  )
})

