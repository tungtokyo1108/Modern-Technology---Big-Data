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
import com.google.cloud.bigquery.BigQueryException;
import java.util.Map;


