"use client";

import {
  KeyboardEvent,
  useEffect,
  useRef,
  useState,
  type FC,
} from "react";

import { getSupportedRoles, type RoleInfo } from "@/lib/roleAnalysis";

export interface SearchableRoleSelectorProps {
  value: string;
  onChange: (value: string) => void;
  onSelect: (role: RoleInfo) => void;
  onClear: () => void;
  placeholder?: string;
  disabled?: boolean;
  error?: string | undefined;
}

interface LoadedRoles {
  roles: RoleInfo[];
  loaded: boolean;
  error: string | null;
}

function filterRoles(roles: RoleInfo[], query: string): RoleInfo[] {
  if (!query.trim()) return roles.slice(0, 8);
  const needle = query.toLowerCase().trim();
  const results = roles.filter((r) => {
    if (r.title.toLowerCase().includes(needle)) return true;
    return r.aliases.some((a) => a.toLowerCase().includes(needle));
  });
  return results.slice(0, 10);
}

export const SearchableRoleSelector: FC<SearchableRoleSelectorProps> = ({
  value,
  onChange,
  onSelect,
  onClear,
  placeholder = "Search for a job role...",
  disabled = false,
  error,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [highlightedIdx, setHighlightedIdx] = useState(-1);
  const [rolesData, setRolesData] = useState<LoadedRoles>({
    roles: [],
    loaded: false,
    error: null,
  });

  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const filtered = filterRoles(rolesData.roles, value);

  useEffect(() => {
    if (!rolesData.loaded && !rolesData.error) {
      getSupportedRoles()
        .then((roles) =>
          setRolesData({ roles, loaded: true, error: null }),
        )
        .catch((err: Error) =>
          setRolesData({
            roles: [],
            loaded: true,
            error: err.message,
          }),
        );
    }
  }, [rolesData]);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
        setHighlightedIdx(-1);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (!isOpen && event.key === "ArrowDown" && filtered.length > 0) {
      event.preventDefault();
      setIsOpen(true);
      setHighlightedIdx(-1);
      return;
    }
    if (!isOpen) {
      if (event.key === "ArrowDown" && filtered.length > 0) {
        event.preventDefault();
        setIsOpen(true);
        setHighlightedIdx(-1);
      }
      return;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      setIsOpen(false);
      setHighlightedIdx(-1);
      return;
    }
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setHighlightedIdx((prev) => (prev + 1) % filtered.length);
      return;
    }
    if (event.key === "ArrowUp") {
      event.preventDefault();
      setHighlightedIdx((prev) => (prev - 1 + filtered.length) % filtered.length);
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      if (highlightedIdx >= 0 && highlightedIdx < filtered.length) {
        const selected = filtered[highlightedIdx];
        onChange(selected.title);
        onSelect(selected);
        setIsOpen(false);
      }
    }
  };

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    onChange(event.target.value);
    setHighlightedIdx(-1);
    if (!isOpen && event.target.value.trim()) {
      setIsOpen(true);
    }
  };

  const showUnsupportedHint =
    value.trim().length > 0 &&
    filtered.length === 0 &&
    !rolesData.error &&
    isOpen;

  const allRolesCount = rolesData.roles.length;

  return (
    <div ref={containerRef} className="relative">
      <div className="relative">
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          onFocus={() => setIsOpen(true)}
          onClick={() => setIsOpen(true)}
          placeholder={placeholder}
          disabled={disabled}
          aria-label="Target job role"
          aria-autocomplete="list"
          aria-controls="role-listbox"
          aria-haspopup="listbox"
          className={`w-full rounded-lg border border-slate-300 bg-white px-3 py-2 pl-10 text-sm text-slate-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 ${
            error ? "border-red-300" : ""
          }`}
        />
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth={1.5}
          strokeLinecap="round"
          strokeLinejoin="round"
          className="absolute left-3 top-2.5 h-4 w-4 text-slate-400"
          aria-hidden="true"
        >
          <circle cx="11" cy="11" r="8" />
          <line x1="21" y1="21" x2="16.65" y2="16.65" />
        </svg>
      </div>

      {error ? (
        <div
          role="alert"
          className="mt-2 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700"
        >
          {error}
        </div>
      ) : null}

      {isOpen && (
        <div className="absolute z-10 mt-1 max-h-64 w-full overflow-y-auto rounded-xl border border-slate-200 bg-white shadow-lg">
          {rolesData.error ? (
            <div className="p-3 text-sm text-slate-500">
              Could not load role list: {rolesData.error}. Type a role manually.
            </div>
          ) : filtered.length > 0 ? (
            <ul role="listbox" id="role-listbox" className="py-1 text-sm">
              {filtered.map((role, idx) => (
                <li key={role.title}>
                  <button
                    type="button"
                    onClick={() => {
                      onChange(role.title);
                      onSelect(role);
                      setIsOpen(false);
                    }}
                    onMouseEnter={() => setHighlightedIdx(idx)}
                    className={`w-full px-3 py-2 text-left hover:bg-indigo-50 ${
                      idx === highlightedIdx
                        ? "bg-indigo-100"
                        : ""
                    }`}
                  >
                    <span className="font-medium text-slate-800">{role.title}</span>
                    {role.aliases.length > 0 ? (
                      <span className="mt-0.5 block text-xs text-slate-500">
                        Also known as: {role.aliases.join(", ")}
                      </span>
                    ) : null}
                  </button>
                </li>
              ))}
            </ul>
          ) : null}

          {showUnsupportedHint ? (
            <div className="p-3 text-sm text-slate-500">
              <p>
                No exact match found for {value.trim()}. Role-level analysis is
                unavailable for this role.
              </p>
              <p className="mt-1">
                Tip: upload or paste a specific job description for targeted
                analysis instead.
              </p>
            </div>
          ) : null}

          {!value.trim() && filtered.length === 0 && !rolesData.error ? (
            <div className="p-3 text-xs text-slate-400">
              {allRolesCount > 0
                ? `Showing top ${filtered.length} of ${allRolesCount} supported roles. Start typing to filter.`
                : "Loading supported roles..."}
            </div>
          ) : null}
        </div>
      )}

      {value && (
        <button
          type="button"
          onClick={() => {
            onChange("");
            onClear();
          }}
          className="absolute right-2 top-2 text-xs text-slate-400 hover:text-slate-600"
          aria-label="Clear role"
        >
          ×
        </button>
      )}
    </div>
  );
};