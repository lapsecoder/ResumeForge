import "@testing-library/jest-dom/vitest";

if (typeof globalThis.URL.createObjectURL !== "function") {
  globalThis.URL.createObjectURL = () => "blob:mock-url";
  globalThis.URL.revokeObjectURL = () => {};
}