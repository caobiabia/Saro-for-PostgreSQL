select  count(*) from comments as c,  		votes as v where c.UserId = v.UserId  AND v.VoteTypeId=2;
