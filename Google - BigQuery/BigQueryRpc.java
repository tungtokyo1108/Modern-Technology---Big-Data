/**
 * Google - Big Data Technology
 * https://github.com/googleapis/google-cloud-java/blob/master/google-cloud-clients/google-cloud-bigquery/src/main/java/com/google/cloud/bigquery/spi/v2/BigQueryRpc.java
 *
 *  Created on: Jan 25, 2019
 *  Student (MIG Virtual Developer): Tung Dang
 */

package com.google.cloud.bigquery.mirror;

import com.google.api.core.InternalExtensionOnly;
import com.google.api.services.bigquery.model.Dataset;
import com.google.api.services.bigquery.model.GetQueryResultsResponse;
import com.google.api.services.bigquery.model.Job;
import com.google.api.services.bigquery.model.Table;
import com.google.api.services.bigquery.model.TableDataInsertAllRequest;
import com.google.api.services.bigquery.model.TableDataInsertAllResponse;
import com.google.api.services.bigquery.model.TableDataList;
import com.google.cloud.ServiceRpc;
import com.google.cloud.Tuple;
import java.util.Map;

@InternalExtensionOnly
public interface BigQueryRpc extends ServiceRpc {
    enum Option 
    {
        FIELDS("fields"),
        DELETE_CONTENTS("deleteContents"),
        ALL_DATASETS("all"),
        ALL_USERS("allUsers"),
        MAX_RESULTS("maxResults"),
        PAGE_TOKEN("pageToken"),
        START_INDEX("startIndex"),
        STATE_FILTER("stateFilter"),
        TIMEOUT("timeoutMs");

        private final String value;

        Option(String value) 
        {
            this.value = value;
        }

        public String value() 
        {
            return value;
        }

        @SuppressWarnings("unchecked")
        <T> T get(Map<Option, ?> options) 
        {
            return (T) options.get(this);
        }

        String getString(Map<Option, ?> options)
        {
            return get(options);
        }

        Long getLong(Map<Option, ?> options)
        {
            return get(options);
        }

        Boolean getBoolean(Map<Option, ?> options)
        {
            return get(options);
        }
    }

    Dataset getDataset(String projectId, String datasetId, Map<Option, ?> options);

    Tuple<String, Iterable<Dataset>> listDatasets(String projectId, Map<Option, ?> options);

    Dataset create(Dataset dataset, Map<Option, ?> options);
    Table create(Table table, Map<Option, ?> options);
    Job create(Job job, Map<Option, ?> options);

    boolean deleteDataset(String projectId, String dataId, Map<Option, ?> options);

    /**
     * Updata information for dataset and table
     */
    Dataset patch(Dataset dataset, Map<Option, ?> options);
    Table patch(Table table, Map<Option, ?> options);
    Table getTable(String projectId, String datasetId, String tableId, Map<Option, ?> options);
    
    Tuple<String, Iterable<Table>> listTable (
        String projectId, String dataset, Map<Option, ?> options
    );

    boolean deleteTable(String projectId, String datasetId, String tableId);
    TableDataInsertAllResponse insertAll(
        String projectId, String datasetId, String tableId, TableDataInsertAllRequest request
    );

    TableDataList listTableData(
        String projectId, String dataId, String tableId, Map<Option, ?> options
    );

    Job getJob(String projectId, String jobId, String location, Map<Option, ?> options);

    Tuple<String, Iterable<Job>> listJobs(String projectId, Map<Option, ?> options);

    boolean cancel(String projectId, String jobId, String location);

    GetQueryResultsResponse getQueryResults(
        String projectId, String jobId, String location, Map<Option, ?> options
    );

    String open(Job job);

    Job write (
        String uploadId, 
        byte[] toWrite,
        int toWriteOffset,
        long destOffset,
        int length,
        boolean last
    );
}
