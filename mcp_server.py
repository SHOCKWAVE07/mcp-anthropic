from mcp.server.fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP("DocumentMCP", log_level="ERROR")


docs = {
    "deposition.md": "This deposition covers the testimony of Angela Smith, P.E.",
    "report.pdf": "The report details the state of a 20m condenser tower.",
    "financials.docx": "These financials outline the project's budget and expenditures.",
    "outlook.pdf": "This document presents the projected future performance of the system.",
    "plan.md": "The plan outlines the steps for the project's implementation.",
    "spec.txt": "These specifications define the technical requirements for the equipment.",
}



@mcp.tool(
    name = "read_doc_contents",
    description = "Reads the contents of a document given its ID and return it as a string. The document ID is the filename of the document.",
)

def read_document(
    doc_id: str = Field(description="The ID of the document to read.")
):
    if doc_id not in docs:
        raise ValueError(f"Document with ID '{doc_id}' not found.")
    return docs[doc_id]


@mcp.tool(
    name = "edit_doc_contents",
    description = "Edits the contents of a document given its ID and new content. Returns the updated content of the document.",
)
def edit_document(
    doc_id: str = Field(description="The ID of the document to edit."),
    old_content: str = Field(description="The old content of the document."),
    new_content: str = Field(description="The new content to write to the document.")
):
    if doc_id not in docs:
        raise ValueError(f"Document with ID '{doc_id}' not found.")

    docs[doc_id] = docs[doc_id].replace(old_content, new_content)
    return docs[doc_id]

# TODO: Write a resource to return all doc id's
# TODO: Write a resource to return the contents of a particular doc
# TODO: Write a prompt to rewrite a doc in markdown format
# TODO: Write a prompt to summarize a doc


if __name__ == "__main__":
    mcp.run(transport="stdio")
