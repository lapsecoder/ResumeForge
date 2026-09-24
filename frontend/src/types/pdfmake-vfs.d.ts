/**
 * pdfmake ships its browser fonts as a plain CJS module (vfs_fonts.js) with
 * no type declarations. It exports a map of base64-encoded TTF files keyed by
 * file name.
 */
declare module "pdfmake/build/vfs_fonts" {
  const vfs: Record<string, string>;
  export default vfs;
}