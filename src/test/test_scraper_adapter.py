import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from infrastructure.scraper.base import ScraperTask
from infrastructure.scraper.producer_adapter import ScraperProducerAdapter
from dependencies.dependencies import get_scraper_client


def test_scraper_task_serialization():
    task = ScraperTask(
        url="https://www.sec.gov/Archives/edgar/data/19617/000119312523056441/d628011d8k.htm",
        parser_type="static_css",
        selectors={"items_text": "body"},
        priority=5,
        max_retries=3,
        context={"ticker": "JPM", "filing_date": "2023-03-01"},
    )
    d = task.to_dict()
    assert d["url"] == task.url
    assert d["parser_type"] == "static_css"
    assert d["parser_config"]["selectors"] == {"items_text": "body"}
    assert d["priority"] == 5
    assert d["max_retries"] == 3
    assert d["context"]["ticker"] == "JPM"


@pytest.mark.asyncio
async def test_scraper_producer_adapter_empty_tasks():
    adapter = ScraperProducerAdapter(base_url="https://mock-scraper.com")
    result = await adapter.dispatch_tasks(job_id="empty-job", tasks=[])
    assert result["status"] == "SKIPPED"
    assert result["total_tasks"] == 0
    assert result["total_batches"] == 0


@pytest.mark.asyncio
async def test_scraper_producer_adapter_batching_and_auth():
    # 5 tareas con batch_size=2 -> 3 batches (2, 2, 1)
    tasks = [
        ScraperTask(url=f"https://sec.gov/doc_{i}.htm", context={"idx": i})
        for i in range(5)
    ]

    adapter = ScraperProducerAdapter(
        base_url="https://webscraping-distributed.duckdns.org",
        auth_user="test_user",
        auth_password="test_password",
    )

    mock_resp = AsyncMock()
    mock_resp.status = 202
    mock_resp.json = AsyncMock(return_value={"job_id": "test-job", "status": "accepted"})

    mock_post_cm = MagicMock()
    mock_post_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_post_cm.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post.return_value = mock_post_cm

    mock_session_cm = MagicMock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_cm.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession", return_value=mock_session_cm):
        result = await adapter.dispatch_tasks(
            job_id="test-job",
            tasks=tasks,
            batch_size=2,
            global_context={"env": "test"},
        )

        assert result["status"] == "SUCCESS"
        assert result["total_tasks"] == 5
        assert result["total_batches"] == 3
        assert result["accepted_tasks"] == 5
        assert len(result["batches"]) == 3
        assert mock_session.post.call_count == 3


@pytest.mark.asyncio
async def test_scraper_producer_adapter_error_handling():
    tasks = [ScraperTask(url="https://sec.gov/doc_fail.htm")]
    adapter = ScraperProducerAdapter(base_url="https://webscraping-distributed.duckdns.org")

    mock_resp = AsyncMock()
    mock_resp.status = 401
    mock_resp.text = AsyncMock(return_value="Unauthorized")

    mock_post_cm = MagicMock()
    mock_post_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_post_cm.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post.return_value = mock_post_cm

    mock_session_cm = MagicMock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_cm.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession", return_value=mock_session_cm):
        with pytest.raises(RuntimeError) as exc_info:
            await adapter.dispatch_tasks(job_id="fail-job", tasks=tasks)
        assert "HTTP 401" in str(exc_info.value)


@pytest.mark.asyncio
async def test_scraper_producer_adapter_retry_success():
    """Verifica que el adaptador reintente ante un error transitorio 500 y tenga éxito en el siguiente intento."""
    tasks = [ScraperTask(url="https://sec.gov/doc_transient.htm")]
    adapter = ScraperProducerAdapter(
        base_url="https://webscraping-distributed.duckdns.org",
        max_batch_retries=3,
        backoff_factor=0.01,  # Rápido para test
    )

    resp_fail = AsyncMock()
    resp_fail.status = 500
    resp_fail.text = AsyncMock(return_value="Internal Server Error")

    resp_ok = AsyncMock()
    resp_ok.status = 202
    resp_ok.json = AsyncMock(return_value={"job_id": "retry-job", "status": "accepted"})

    cm_fail = MagicMock()
    cm_fail.__aenter__ = AsyncMock(return_value=resp_fail)
    cm_fail.__aexit__ = AsyncMock(return_value=None)

    cm_ok = MagicMock()
    cm_ok.__aenter__ = AsyncMock(return_value=resp_ok)
    cm_ok.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post.side_effect = [cm_fail, cm_ok]

    mock_session_cm = MagicMock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_cm.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession", return_value=mock_session_cm):
        result = await adapter.dispatch_tasks(job_id="retry-job", tasks=tasks)

        assert result["status"] == "SUCCESS"
        assert result["accepted_tasks"] == 1
        assert result["batches"][0]["attempts"] == 2
        assert mock_session.post.call_count == 2


def test_dependency_injection_scraper_client():
    client = get_scraper_client()
    assert isinstance(client, ScraperProducerAdapter)
