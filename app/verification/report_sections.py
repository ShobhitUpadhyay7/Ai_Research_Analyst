from app.verification.schema import CitationVerification, Conflict


def build_citation_verification_section(
    verifications: list[CitationVerification],
) -> str:
    lines = [
        "## Citation Verification",
        "",
    ]

    if not verifications:
        lines.append("No citations were verified.")
        return "\n".join(lines)

    for verification in verifications:
        if verification.supported:
            icon = "✅"
        elif verification.status == "not_cited":
            icon = "ℹ️"
        else:
            icon = "⚠️"

        url_text = ""

        if verification.url_reachable is True:
            url_text = " | URL reachable"
        elif verification.url_reachable is False:
            url_text = " | URL unreachable"

        lines.append(
            f"- {icon} [{verification.citation_key}] "
            f"{verification.status} ({verification.confidence}) — "
            f"{verification.reason}{url_text}"
        )

    return "\n".join(lines)


def build_verified_conflicts_section(
    conflicts: list[Conflict],
) -> str:
    lines = [
        "## Verified Conflicts",
        "",
    ]

    if not conflicts:
        lines.append("No conflicts were detected from the selected evidence.")
        return "\n".join(lines)

    for conflict in conflicts:
        if conflict.citation_keys:
            keys = ", ".join(
                f"[{key}]"
                for key in conflict.citation_keys
            )
        else:
            keys = "[No citation keys]"

        lines.append(
            f"- ({conflict.severity}) {conflict.description} {keys}"
        )

    return "\n".join(lines)