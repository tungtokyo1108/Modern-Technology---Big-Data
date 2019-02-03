/**
 * Google - Big Data Technology
 * https://github.com/googleapis/google-cloud-java/blob/master/google-cloud-clients/google-cloud-bigquery/src/main/java/com/google/cloud/bigquery/spi/v2/HttpBigQueryRpc.java
 *
 *  Created on: Jan 29, 2019
 *  Student (MIG Virtual Developer): Tung Dang
 */

package com.google.cloud.bigquery.mirror;

import static java.net.HttpURLConnection.HTTP_CREATED;
import static java.net.HttpURLConnection.HTTP_NOT_FOUND;
import static java.net.HttpURLConnection.HTTP_OK;

import com.google.api.client.http.ByteArrayContent;
import com.google.api.client.http.GenericUrl;
import com.google.api.client.http.HttpRequest;
import com.google.api.client.http.HttpRequestFactory;
import com.google.api.client.http.HttpRequestInitializer;
import com.google.api.client.http.HttpResponse;
import com.google.api.client.http.HttpResponseException;
import com.google.api.client.http.HttpTransport;
import com.google.api.client.http.json.JsonHttpContent;
import com.google.api.client.json.JsonFactory;
import com.google.api.client.json.jackson2.JacksonFactory;
import com.google.api.core.InternalApi;
import com.google.api.core.InternalExtensionOnly;
import com.google.api.services.bigquery.Bigquery;
import com.google.api.services.bigquery.model.Dataset;
import com.google.api.services.bigquery.model.DatasetList;
import com.google.api.services.bigquery.model.DatasetReference;
import com.google.api.services.bigquery.model.GetQueryResultsResponse;
import com.google.api.services.bigquery.model.Job;
import com.google.api.services.bigquery.model.JobList;
import com.google.api.services.bigquery.model.JobStatus;
import com.google.api.services.bigquery.model.Table;
import com.google.api.services.bigquery.model.TableDataInsertAllRequest;
import com.google.api.services.bigquery.model.TableDataInsertAllResponse;
import com.google.api.services.bigquery.model.TableDataList;
import com.google.api.services.bigquery.model.TableList;
import com.google.api.services.bigquery.model.TableReference;
import com.google.cloud.Tuple;
import com.google.cloud.bigquery.BigQueryException;
import com.google.cloud.bigquery.BigQueryOptions;
import com.google.cloud.http.HttpTransportOptions;
import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import java.io.IOException;
import java.math.BigInteger;
import java.util.List;
import java.util.Map;

import javax.sound.sampled.DataLine;

@InternalExtensionOnly
public class HttpBigQueryRpc implements BigQueryRpc 
{
    public static final String DEFAULT_PROJECTION = "full";
    private static final String BASE_RESUMABLE_URI = 
        "https://www.googleapis.com/upload/bigquery/v2/projects/";
    private static final int HTTP_RESUME_INCOMPLETE = 308;
    private final BigQueryOptions options;
    private final Bigquery bigquery;

    @InternalApi("Visiable for testing")
    static final Function<DatasetList.Datasets, Dataset> LIST_TO_DATASET = 
        new Function<DatasetList.Datasets,Dataset>() {
            @Override
            public Dataset apply(DatasetList.Datasets datasetPb) {
                return new Dataset()
                    .setDatasetReference(datasetPb.getDatasetReference())
                    .setFriendlyName(datasetPb.getFriendlyName())
                    .setId(datasetPb.getId())
                    .setKind(datasetPb.getKind())
                    .setLabels(datasetPb.getLabels());
            }
        }; 

    public HttpBigQueryRpc(BigQueryOptions options) 
    {
        HttpTransportOptions transportOptions = (HttpTransportOptions) options.getTransportOptions();
        HttpTransport transport = transportOptions.getHttpTransportFactory().create();
        HttpRequestInitializer initializer = transportOptions.getHttpRequestInitializer(options);
        this.options = options;
        bigquery = 
            new Bigquery.Builder(transport, new JacksonFactory(), initializer)
                .setRootUrl(options.getHost())
                .setApplicationName(options.getApplicationName())
                .build();
    }

    private static BigQueryException translate(IOException exception)
    {
        return new BigQueryException(exception);
    }

    @Override 
    public Dataset getDataset(String projectId, String datasetId, Map<Option, ?> options)
    {
        try {
            return bigquery
                .datasets()
                .get(projectId, datasetId)
                .setFields(Option.FIELDS.getString(options))
                .execute();
        } catch (IOException ex) {
            BigQueryException serviceException = translate(ex);
            if (serviceException.getCode() == HTTP_NOT_FOUND)
            {
                return null;
            }
            throw serviceException;
        }
    }

    @Override
    public Tuple <String, Iterable<Dataset>> listDatasets(String projectId, Map<Option, ?> options)
    {
        try {
            DatasetList datasetList = 
                bigquery 
                    .datasets()
                    .list(projectId)
                    .setAll(Option.ALL_DATASETS.getBoolean(options))
                    .setMaxResults(Option.MAX_RESULTS.getLong(options))
                    .setPageToken(Option.PAGE_TOKEN.getString(options))
                    .setPageToken(Option.PAGE_TOKEN.getString(options))
                    .execute();
            Iterable<DatasetList.Datasets> datasets = datasetList.getDatasets();
            return Tuple.of(
                datasetList.getNextPageToken(),
                Iterables.transform(datasets != null ? datasets : ImmutableList.<DatasetList.Datasets>of(),
                LIST_TO_DATASET)
            );
        } catch (IOException ex) {
            throw translate(ex);
        }
    }

    @Override
    public Dataset create(Dataset dataset, Map<Option, ?> options)
    {
        try {
            return bigquery 
                .datasets()
                .insert(dataset.getDatasetReference().getProjectId(), dataset)
                .setFields(Option.FIELDS.getString(options))
                .execute();
        } catch (IOException ex) {
            throw translate(ex);
        }
    }

    @Override
    public Table create(Table table, Map<Option, ?> options) 
    {
        try {
            table.setType(null);
            TableReference reference = table.getTableReference();
            return bigquery 
                .tables()
                .insert(reference.getProjectId(), reference.getDatasetId(), table)
                .setFields(Option.FIELDS.getString(options))
                .execute();
        } catch (IOException ex) {
            throw translate(ex);
        }
    }

    @Override
    public Job create(Job job, Map<Option, ?> options) 
    {
        try {
            String projectId = 
                job.getJobReference() != null 
                    ? job.getJobReference().getProjectId()
                    : this.options.getProjectId();
            return bigquery 
                .jobs()
                .insert(projectId, job)
                .setFields(Option.FIELDS.getString(options))
                .execute();
        } catch (IOException ex) {
            throw translate(ex);
        }
    }

    @Override
    public boolean deleteDataset(String projectId, String datasetId, Map<Option, ?> options) 
    {
        try {
            bigquery
                .datasets()
                .delete(projectId, datasetId)
                .setDeleteContents(Option.DELETE_CONTENTS.getBoolean(options))
                .execute();
            return true;
        } catch (IOException ex) {
            BigQueryException serviceException = translate(ex);
            if (serviceException.getCode() == HTTP_NOT_FOUND) 
            {
                return false;
            }
            throw serviceException;
        }
    }

    @Override
    public Dataset patch(Dataset dataset, Map<Option, ?> options)
    {
        try {
            DatasetReference reference = dataset.getDatasetReference();
            return bigquery 
                .datasets()
                .patch(reference.getProjectId(), reference.getDatasetId(), dataset)
                .setFields(Option.FIELDS.getString(options))
                .execute();
        } catch (IOException ex) {
            throw translate(ex);
        }
    }

    @Override 
    public Table patch(Table table, Map<Option, ?> options)
    {
        try {
            table.setType(null);
            TableReference reference = table.getTableReference();
            return bigquery 
                .tables()
                .patch(reference.getProjectId(), reference.getDatasetId(), reference.getTableId(), table)
                .setFields(Option.FIELDS.getString(options))
                .execute();
        } catch (IOException ex) {
            throw translate(ex);
        }
    }

    @Override
    public Table getTable (String projectId, String datasetId, String tableId, Map<Option, ?> options)
    {
        try {
            return bigquery 
                .tables()
                .get(projectId, datasetId, tableId)
                .setFields(Option.FIELDS.getString(options))
                .execute();
        } catch (IOException ex) {
            BigQueryException serviceException = translate(ex);
            if (serviceException.getCode() == HTTP_NOT_FOUND)
            {
                return null;
            }
            throw serviceException;
        }
    }

    @Override
    public Tuple<String, Iterable<Table>> listTables(String projectId, String datasetId, Map<Option, ?> options) 
    {
        try {
            TableList tableList = 
                bigquery 
                    .tables()
                    .list(projectId, datasetId)
                    .setMaxResults(Option.MAX_RESULTS.getLong(options))
                    .setPageToken(Option.PAGE_TOKEN.getString(options))
                    .execute();
            Iterable<TableList.Tables> tables = tableList.getTables();
            return Tuple.of(
                tableList.getNextPageToken(),
                Iterables.transform(
                    tables != null ? tables : ImmutableList.<TableList.Tables>of(),
                    new Function<TableList.Tables, Table>() {
                        @Override
                        public Table apply(TableList.Tables tablePb) {
                            return new Table()
                                .setFriendlyName(tablePb.getFriendlyName())
                                .setId(tablePb.getId())
                                .setKind(tablePb.getKind())
                                .setTableReference(tablePb.getTableReference())
                                .setType(tablePb.getType());
                        }
                    }
                )
            );
        } catch (IOException ex) {
            throw translate(ex);
        }
    }
}
