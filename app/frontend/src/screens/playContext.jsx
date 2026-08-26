import React, { createContext, useCallback, useContext } from "react";

export const PlayCtx = createContext(null);
export const usePlay = () => useContext(PlayCtx);

export function fileToDataUrl(file) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(r.result);
    r.onerror = rej;
    r.readAsDataURL(file);
  });
}

/** `useState` that lives in the Play provider instead of the component.
 *
 *  Play unmounts the inactive tab, so anything a long-running job needs to
 *  survive — its id, its progress, its results — cannot live in that tab's own
 *  state. Same signature as useState, so call sites barely change.
 */
export function usePlayState(key, initial) {
  const { tabState, setTabStateKey } = usePlay();
  const value = tabState && key in tabState ? tabState[key] : initial;
  const set = useCallback(
    (v) => setTabStateKey(key, v, initial),
    [key, setTabStateKey, initial],
  );
  return [value, set];
}
