/**
 * Phase 7D — template-facing helpers for safely rendering untrusted resume
 * text. React escapes text nodes automatically; these helpers additionally
 * guarantee that only http(s)/mailto links ever reach an `href`, so values
 * like `javascript:` are rendered as inert text.
 */

import type { ContactInfo } from "@/lib/resume";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const BARE_DOMAIN_RE = /^[a-z0-9-]+(\.[a-z0-9-]+)+(\/[^\s]*)?$/i;

export function safeProfileHref(value?: string | null): string | null {
  const raw = value?.trim();
  if (!raw) return null;
  if (/^https?:\/\/[^\s]+$/i.test(raw)) return raw;
  if (BARE_DOMAIN_RE.test(raw)) return `https://${raw}`;
  return null;
}

export function safeEmailHref(value?: string | null): string | null {
  const raw = value?.trim();
  if (!raw || !EMAIL_RE.test(raw)) return null;
  return `mailto:${raw}`;
}

export function dateRange(start?: string | null, end?: string | null): string {
  return [start, end].map((part) => part?.trim()).filter(Boolean).join(" – ");
}

export function isBlank(value?: string | null): boolean {
  return !value || !value.trim();
}

export interface ContactItem {
  key: string;
  value: string;
  href?: string;
}

export function contactItems(contact: ContactInfo | undefined): ContactItem[] {
  if (!contact) return [];
  const items: ContactItem[] = [];
  const addText = (key: string, value?: string | null) => {
    if (!isBlank(value)) items.push({ key, value: value!.trim() });
  };
  const addLink = (key: string, value?: string | null) => {
    if (isBlank(value)) return;
    const href = safeProfileHref(value);
    items.push({ key, value: value!.trim(), href: href ?? undefined });
  };
  const emailHref = safeEmailHref(contact.email);
  if (!isBlank(contact.email)) {
    items.push({ key: "email", value: contact.email!.trim(), href: emailHref ?? undefined });
  }
  addText("phone", contact.phone);
  addText("location", contact.location);
  addLink("linkedin", contact.linkedin);
  addLink("github", contact.github);
  addLink("website", contact.website);
  return items;
}

export function hasAnyContact(contact: ContactInfo | undefined): boolean {
  return contactItems(contact).length > 0;
}
