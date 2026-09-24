import { describe, expect, it } from "vitest";

import {
  commitHistory,
  createHistory,
  HISTORY_LIMIT,
  redoHistory,
  resetHistory,
  undoHistory,
} from "./history";

describe("history reducer", () => {
  it("starts empty with the initial present", () => {
    const state = createHistory("a");
    expect(state).toEqual({ past: [], present: "a", future: [], mergeKey: null });
  });

  it("commits, undoes, and redoes", () => {
    let state = createHistory("a");
    state = commitHistory(state, "b");
    state = commitHistory(state, "c");
    expect(state.present).toBe("c");
    expect(state.past).toEqual(["a", "b"]);

    state = undoHistory(state);
    expect(state.present).toBe("b");
    state = undoHistory(state);
    expect(state.present).toBe("a");
    expect(undoHistory(state).present).toBe("a");

    state = redoHistory(state);
    expect(state.present).toBe("b");
    state = redoHistory(state);
    expect(state.present).toBe("c");
    expect(redoHistory(state).present).toBe("c");
  });

  it("clears the redo stack when a new change is committed", () => {
    let state = createHistory("a");
    state = commitHistory(state, "b");
    state = undoHistory(state);
    expect(state.future).toEqual(["b"]);
    state = commitHistory(state, "c");
    expect(state.future).toEqual([]);
    expect(redoHistory(state).present).toBe("c");
  });

  it("ignores an unchanged present", () => {
    const state = createHistory("a");
    expect(commitHistory(state, "a")).toBe(state);
  });

  it("merges consecutive commits that share a mergeKey", () => {
    let state = createHistory("");
    state = commitHistory(state, "a", { mergeKey: "name" });
    state = commitHistory(state, "ab", { mergeKey: "name" });
    state = commitHistory(state, "abc", { mergeKey: "name" });
    expect(state.past).toEqual([""]);
    state = undoHistory(state);
    expect(state.present).toBe("");
  });

  it("starts a new step when the mergeKey changes", () => {
    let state = createHistory("");
    state = commitHistory(state, "a", { mergeKey: "name" });
    state = commitHistory(state, "b", { mergeKey: "summary" });
    expect(state.past).toEqual(["", "a"]);
  });

  it("bounds the past to HISTORY_LIMIT", () => {
    let state = createHistory(0);
    for (let i = 1; i <= HISTORY_LIMIT + 20; i += 1) {
      state = commitHistory(state, i);
    }
    expect(state.past).toHaveLength(HISTORY_LIMIT);
    expect(state.present).toBe(HISTORY_LIMIT + 20);
  });

  it("reset discards past and future", () => {
    let state = createHistory("a");
    state = commitHistory(state, "b");
    state = undoHistory(state);
    state = resetHistory("z");
    expect(state).toEqual({ past: [], present: "z", future: [], mergeKey: null });
  });
});
