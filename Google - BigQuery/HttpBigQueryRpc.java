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

}
