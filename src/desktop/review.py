"""Keep the actual review decision separate from measurements and processing notes."""


def explanation(photo):
    if photo.recycled or photo.status == 'Pending':
        return '', ''
    reasons = photo.review_reasons
    if reasons is None:
        reasons = [reason for reason in photo.reasons if not reason.startswith((
            'RAW assessment uses', 'RAW focus was checked', 'Learned Keep score:',
            'The learned score estimates'))]
    if photo.status in ('Error', 'Unsupported'):
        return 'Assessment unavailable', '\n'.join(reasons) or 'This file could not be assessed. Inspect the file or try analysis again.'
    if photo.status == 'Review':
        title = 'Why this photo needs review'
        if photo.override == 'Keep':
            title = 'AI review reasons — approved by you'
        text = '\n'.join('• '+reason for reason in reasons)
        return title, text or 'The saved assessment requested Review but did not record a specific reason.'
    if photo.override == 'Review':
        return 'Marked for review by you', 'You marked this photo for manual review. The automatic assessment was Keep.'
    return '', ''
