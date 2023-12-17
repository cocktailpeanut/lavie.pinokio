module.exports = {
  title: "lavie",
  description: "Text-to-Video (T2V) generation framework from Vchitect https://github.com/Vchitect/LaVie",
  icon: "icon.jpeg",
  menu: async (kernel) => {
    let installing = await kernel.running(__dirname, "install.json")
    let installed = await kernel.exists(__dirname, "env")
    let running = await kernel.running(__dirname, "start.json")
    if (installing) {
      return [{
        icon: "fa-solid fa-plug",
        text: "Installing",
        href: "install.json",
        params: { fullscreen: true }
      }]
    } else if (installed) {
      if (running) {
        let session = await kernel.require(__dirname, "session.json")
        if (session && session.url) {
          return [{
            icon: 'fa-solid fa-spin fa-circle-notch',
            text: "Running",
            type: "label"
          }, {
            icon: 'fa-solid fa-terminal',
            text: "Terminal",
            href: "start.json",
            params: { fullscreen: true }
          }, {
            icon: "fa-solid fa-rocket",
            text: "Open Web UI",
            href: session.url,
            target: "_blank"
          }]
        } else {
          return [{
            icon: 'fa-solid fa-spin fa-circle-notch',
            text: "Running",
            type: "label"
          }, {
            icon: 'fa-solid fa-terminal',
            text: "Terminal",
            href: "start.json",
            params: { fullscreen: true }
          }]
        }
      } else {
        return [{
          icon: "fa-solid fa-power-off",
          text: "Start",
          href: "start.json",
          params: { fullscreen: true, run: true }
        }]
      }
    } else {
      return [{
        icon: "fa-solid fa-plug",
        text: "Install",
        href: "install.json",
        params: { run: true, fullscreen: true }
      }]
    }
  }
}
