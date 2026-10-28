import { labTitle, researchDisclaimer } from "../lib/site";

export default function HomePage() {
  return (
    <main>
      <h1>{labTitle}</h1>
      <p>AFIL V1 skeleton. The API listens on port 8000.</p>
      <p>{researchDisclaimer}</p>
    </main>
  );
}
