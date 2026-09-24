import type { Resume } from "./resume";

/** A fully-populated resume used by PDF document and render tests. */
export function makeFixtureResume(): Resume {
  return {
    contact: {
      name: "Ada Lovelace",
      email: "ada@example.com",
      phone: "+44 20 7946 0958",
      location: "London, UK",
      linkedin: "linkedin.com/in/ada",
    },
    summary: "Mathematician and first programmer, pioneer of the analytical engine.",
    experience: [
      {
        company: "Analytical Engines Ltd",
        title: "Lead Engineer",
        start_date: "1843-01",
        end_date: "1848-12",
        location: "London, UK",
        description: "Designed algorithms for a general-purpose computing machine.",
        achievements: ["Published extensive notes", "Devised the first algorithm"],
      },
      {
        company: "Royal Institute",
        title: "Collaborator",
        start_date: "1838-01",
        end_date: "1842-06",
        description: "Translated and annotated works on punched-card machinery.",
        achievements: [],
      },
    ],
    education: [
      {
        institution: "Home Academy",
        degree: "BSc",
        field: "Mathematics",
        start_date: "1834-01",
        end_date: "1840-01",
        details: ["Studied geometry", "Studied symbolic logic"],
      },
    ],
    projects: [
      {
        name: "Note G",
        description: "First published algorithm, a plan for computing Bernoulli numbers.",
        technologies: ["Punched cards", "Difference engine"],
      },
    ],
    skills: {
      technical: ["Algorithm design", "Mathematics"],
      soft: ["Analytical reasoning", "Collaboration"],
      tools: ["Analytical engine"],
      languages: ["English", "French", "Italian"],
    },
    certifications: [{ name: "Fellow", issuer: "Royal Society", date: "1841" }],
    custom_sections: [{ heading: "Awards", content: ["Order of Merit", "Best notes on machinery"] }],
    metadata: { overall_confidence: "high", word_count: 42 },
  };
}

/** A resume long enough to force multiple A4 pages. */
export function makeLongResume(): Resume {
  const resume = makeFixtureResume();
  resume.experience = Array.from({ length: 30 }, (_, index) => ({
    company: `Corp ${index}`,
    title: `Role ${index}`,
    start_date: `18${index % 50 + 40}-01`,
    end_date: `18${index % 50 + 41}-12`,
    description:
      "Responsible for designing, implementing and shipping reliable computing machinery across " +
      "several departments. Led cross-functional teams and documented every step of the engineering " +
      "process for future reference by colleagues and successors.",
    achievements: [
      `Delivered milestone ${index}`,
      `Published report ${index}`,
      "Improved process efficiency",
    ],
  }));
  return resume;
}