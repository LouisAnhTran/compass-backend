"""LLM contracts for the two extraction nodes (SPEC §7).

Both preserve the user's phrasing rather than resolving to canonical names —
resolution is the job of the fuzzy ranker plus a human confirmation, and an LLM
that "helpfully" rewrites "invoicing" to "Q2 - Invoicing EU" destroys the
evidence the HITL shortlist is built from.
"""

EXTRACT_MODEL = """\
You extract the document type / model the user is asking about.

Rules:
- Return the user's own phrasing. Do NOT expand, correct, or canonicalise it.
  "receipts" stays "receipts", not "Receipt" and not "Official Receipt".
- Strip plurals only where the singular is obviously the same word
  ("receipts" -> "receipt"). Never guess at anything more.
- If the user mentioned no document type at all, return an empty list.
- At most one entry: this system searches one model at a time.

User query:
{query}
"""

EXTRACT_OTHER = """\
You extract search filters from a document-search query.

Today's date is {today} (dd/mm/yyyy). Use it to resolve relative dates.

Extract three things:

1. queues — every queue mention. A queue may be named ("Invoices EU"), coded
   ("Q1"), or given as a bare numeric id ("2321"). Keep numeric ids as strings.
   Return the user's phrasing unchanged. Empty list if none mentioned.

2. members — every person mention: first name, full name, or an email
   fragment. Return as written. Empty list if none mentioned.

3. date_range — resolve any date expression to {{"from": "dd/mm/yyyy",
   "to": "dd/mm/yyyy"}}. Both bounds are required and inclusive.
     "last week"        -> the previous Mon-Sun window
     "September"        -> 01/09 to the last day of that September
     "01/09 to 14/09"   -> exactly that, current year unless stated
     "today"            -> {today} to {today}
   If no date was mentioned, return null.

Do not invent filters the user did not ask for. An absent filter is an empty
list or null, never a guess.

User query:
{query}
"""
