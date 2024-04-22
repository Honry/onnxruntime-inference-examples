function loadScript(scriptName) {
  const s = document.createElement('script');
  s.src = chrome.runtime.getURL(scriptName);
  const extensionId = chrome.runtime.id;
  s.dataset.params = JSON.stringify({extensionId});
  console.log(`Loads ${scriptName} from extension ID: ${extensionId}`);
  s.onload = function() { this.remove(); };
  (document.head || document.documentElement).appendChild(s);
}
loadScript('polyfill.js');