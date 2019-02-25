/**
 * Google - Big Data Technology
 * https://github.com/googleapis/google-cloud-java/blob/master/google-cloud-clients/google-cloud-bigquery/src/main/java/com/google/cloud/bigquery/InsertAllResponse.java
 *
 *  Created on: Feb 22, 2019
 *  Student (MIG Virtual Developer): Tung Dang
 */

package com.google.cloud.bigquery.mirror;

import com.google.api.services.bigquery.model.ErrorProto;
import com.google.api.services.bigquery.model.TableDataInsertAllResponse;
import com.google.api.services.bigquery.model.TableDataInsertAllResponse.InsertErrors;
import com.google.cloud.bigquery.BigQueryError;
import com.google.common.base.Function;
import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import java.io.Serializable;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public class InsertAllResponse implements Serializable
{
    private static final long serialVersionUID = -6934152676514098452L;

    private final Map<Long, List<BigQueryError>> insertErrors;

    InsertAllResponse(Map<Long, List<BigQueryError>> insertErrors)
    {
        this.insertErrors = 
            insertErrors != null
                ? ImmutableMap.copyOf(insertErrors)
                : ImmutableMap.<Long, List<BigQueryError>>of();
    }

    public Map<Long, List<BigQueryError>> getInsertErrors()
    {
        return insertErrors;
    }

    public List<BigQueryError> getErrorsFor(long index)
    {
        return insertErrors.get(index);
    }

    public boolean hasErrors()
    {
        return !insertErrors.isEmpty();
    }

    @Override
    public final int hashCode()
    {
        return Objects.hash(insertErrors);
    }

    @Override
    public final boolean equals(Object obj)
    {
        return obj == this
            || obj != null 
                && obj.getClass().equals(InsertAllResponse.class)
                && Objects.equals(insertErrors, ((InsertAllResponse) obj).insertErrors);
    }

    @Override
    public String toString()
    {
        return MoreObjects.toStringHelper(this).add("insertErrors", insertErrors).toString();
    }

    TableDataInsertAllResponse toPb()
    {
        TableDataInsertAllResponse responsePb = new TableDataInsertAllResponse();
        if (!insertErrors.isEmpty())
        {
            responsePb.setInsertErrors(
                ImmutableList.copyOf(
                    Iterables.transform(
                        insertErrors.entrySet(),
                        new Function<Map.Entry<Long, List<BigQueryError>>, InsertErrors>() {
                            @Override
                            public InsertErrors apply(Map.Entry<Long, List<BigQueryError>> entry) {
                                return new InsertErrors()
                                    .setIndex(entry.getKey())
                                    .setErrors(
                                        Lists.transform(entry.getValue(), BigQueryError.To_PB_FUNCTION));
                            }
                        }
                    )
                )
            );
        }
        return responsePb;
    }

    static InsertAllResponse fromPb(TableDataInsertAllResponse responsePb)
    {
        Map<Long, List<BigQueryError>> insertErrors = null;
        if (responsePb.getInsertErrors() != null)
        {
            List<InsertErrors> errorsPb = responsePb.getInsertErrors();
            insertErrors = Maps.newHashMapWithExpectedSize(errorsPb.size());
            for (InsertErrors errorPb : errorsPb)
            {
                insertErrors.put(
                    errorPb.getIndex(),
                    Lists.transform(
                        errorPb.getErrors() != null ? errorPb.getErrors() : ImmutableList.<ErrorProto>of(),
                        BigQueryError.FROM_PB_FUNCTION
                    )
                );
            }
        }
    }
}
