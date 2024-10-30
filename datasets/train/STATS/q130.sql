select  count(*) from posts as p,          tags as t,          votes as v  where p.Id = t.ExcerptPostId 	and p.OwnerUserId = v.UserId  AND p.CommentCount>=0;
