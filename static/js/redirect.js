/**
 * NewsML — Redirect Page Logic
 * static/js/redirect.js
 *
 * Checks whether the Flask dev server is running on localhost:5000.
 * Shows the correct state card (checking / running / offline) and
 * auto-redirects when the server is found.
 */

"use strict";

const FLASK_URL = "http://localhost:5000";

/**
 * Show one of the three state cards and hide the others.
 * @param {"checking"|"running"|"offline"} state
 */
function showState(state) {
  document.getElementById("state-checking").style.display = state === "checking" ? "block" : "none";
  document.getElementById("state-running").style.display  = state === "running"  ? "block" : "none";
  document.getElementById("state-offline").style.display  = state === "offline"  ? "block" : "none";
}

/**
 * Ping the Flask /model-status endpoint.
 * Uses mode:"no-cors" to avoid a CORS error — we only need to know
 * whether the server responds at all, not read its response body.
 */
async function checkServer() {
  try {
    await fetch(FLASK_URL + "/model-status", {
      signal: AbortSignal.timeout(2000),
      mode: "no-cors",
    });
    // If fetch didn't throw, the server is up
    showState("running");
    setTimeout(() => { window.location.href = FLASK_URL; }, 800);
  } catch {
    showState("offline");
  }
}

// Small delay so the "checking" spinner is visible before the result
setTimeout(checkServer, 800);
