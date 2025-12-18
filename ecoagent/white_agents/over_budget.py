def run(ga):
    _ = ga.request_dataset("target", split="train")
    _ = ga.request_dataset("median_listing_price", split="train")
    _ = ga.request_dataset("median_income", split="train") # disqualified