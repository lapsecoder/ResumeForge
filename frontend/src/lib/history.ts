/**
 * Phase 7D — bounded, in-memory undo/redo history.
 *
 * The history lives only in React state for the current session. It is never
 * written to storage, cookies, or the network, and refreshing the page
 * discards it along with the working resume.
 *
 * Consecutive edits that share a `mergeKey` (for example typing in one field)
 * collapse into a single undo step, so typing does not flood the history.
 */

import { useCallback, useState } from "react";

/** Maximum number of undo steps retained. */
export const HISTORY_LIMIT = 50;

export interface HistoryCommitOptions {
  /** Non-null values merge consecutive commits with the same key. */
  mergeKey?: string | null;
}

export interface History<T> {
  past: T[];
  present: T;
  future: T[];
  mergeKey: string | null;
}

export function createHistory<T>(present: T): History<T> {
  return { past: [], present, future: [], mergeKey: null };
}

/** Record a new present value. No-op when the value is unchanged. */
export function commitHistory<T>(
  state: History<T>,
  present: T,
  options?: HistoryCommitOptions
): History<T> {
  if (Object.is(state.present, present)) return state;
  const mergeKey = options?.mergeKey ?? null;
  if (mergeKey !== null && mergeKey === state.mergeKey) {
    return { past: state.past, present, future: [], mergeKey };
  }
  const past = [...state.past, state.present];
  if (past.length > HISTORY_LIMIT) past.splice(0, past.length - HISTORY_LIMIT);
  return { past, present, future: [], mergeKey };
}

export function undoHistory<T>(state: History<T>): History<T> {
  if (state.past.length === 0) return state;
  const present = state.past[state.past.length - 1];
  return {
    past: state.past.slice(0, -1),
    present,
    future: [state.present, ...state.future],
    mergeKey: null,
  };
}

export function redoHistory<T>(state: History<T>): History<T> {
  if (state.future.length === 0) return state;
  const [present, ...future] = state.future;
  return { past: [...state.past, state.present], present, future, mergeKey: null };
}

/** Discard all history and start again from `present`. */
export function resetHistory<T>(present: T): History<T> {
  return createHistory(present);
}

export interface UseHistoryResult<T> {
  present: T;
  canUndo: boolean;
  canRedo: boolean;
  commit: (present: T, options?: HistoryCommitOptions) => void;
  undo: () => void;
  redo: () => void;
  reset: (present: T) => void;
}

export function useHistory<T>(initial: T): UseHistoryResult<T> {
  const [state, setState] = useState<History<T>>(() => createHistory(initial));

  const commit = useCallback((present: T, options?: HistoryCommitOptions) => {
    setState((prev) => commitHistory(prev, present, options));
  }, []);

  const undo = useCallback(() => setState(undoHistory), []);
  const redo = useCallback(() => setState(redoHistory), []);

  const reset = useCallback((present: T) => {
    setState(() => resetHistory(present));
  }, []);

  return {
    present: state.present,
    canUndo: state.past.length > 0,
    canRedo: state.future.length > 0,
    commit,
    undo,
    redo,
    reset,
  };
}
